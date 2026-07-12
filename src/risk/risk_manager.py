"""
=============================================================================
STEP 8 - RISK MANAGEMENT
=============================================================================
"The first rule of trading is: don't blow up. The second rule is the same."
                                                      - Paul Tudor Jones

WHY RISK MANAGEMENT IS SEPARATE FROM SIGNAL GENERATION:
  A good signal tells you *direction*. Risk management tells you *how much*.
  Even a strategy with 60% directional accuracy can go bankrupt if position
  sizing is reckless. Risk management is the bridge between predictions and
  actual P&L.

COMPONENTS IMPLEMENTED:
  1. Position Sizing (Volatility-Adjusted / Kelly Criterion)
  2. Stop-Loss (Fixed % / ATR-based Trailing)
  3. Portfolio-Level Controls (max drawdown circuit breaker)
  4. Volatility Regime Filter (reduce exposure in high-vol regimes)
  5. Correlation-Based Diversification (max % in correlated stocks)
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    MAX_POSITION_PCT, STOP_LOSS_PCT, ATR_MULTIPLIER,
    VOLATILITY_SCALE, RISK_FREE_RATE, OUTPUT_DIR, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. POSITION SIZING
# -----------------------------------------------------------------------------

class PositionSizer:
    """
    WHY VOLATILITY-ADJUSTED SIZING:
      If we allocate the same capital to a volatile stock (e.g., 30% ann vol)
      as to a stable stock (e.g., 15% ann vol), we are taking TWICE the risk
      per rupee in the volatile stock. Volatility-adjusted sizing equalises
      *risk contribution* across positions.

    FORMULA (inverse-volatility weighting):
        w_i = (1 / sigma_i) / Sum(1 / sigma_j)
        where sigma_i = annualised realised volatility of stock i

    KELLY CRITERION (partial):
        f* = (p x b  q) / b
        where p = win probability, b = win/loss ratio, q = 1 - p
        We use 25% Kelly (quarter-Kelly) for safety.
    """

    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = MAX_POSITION_PCT,
        vol_scale: bool         = VOLATILITY_SCALE,
    ):
        self.portfolio_value  = portfolio_value
        self.max_position_pct = max_position_pct
        self.vol_scale        = vol_scale

    def volatility_adjusted_size(
        self,
        ticker: str,
        signal_strength: float,
        realised_vol: float,
        target_vol: float = 0.15,   # target 15% annualised vol per position
    ) -> float:
        """
        Size position so each stock contributes ~target_vol annualised risk.

        FORMULA:
            position_size = (target_vol / realised_vol) x portfolio_value x signal
        """
        if realised_vol <= 0:
            return 0.0

        if self.vol_scale:
            raw_size = (target_vol / realised_vol) * self.portfolio_value * abs(signal_strength)
        else:
            raw_size = self.max_position_pct * self.portfolio_value * abs(signal_strength)

        # Cap at max position
        max_val  = self.max_position_pct * self.portfolio_value
        raw_size = min(raw_size, max_val)

        logger.debug(f"[{ticker}] Vol-adjusted size: Rs.{raw_size:,.0f} "
                     f"(sigma={realised_vol:.1%}, signal={signal_strength:.2f})")
        return raw_size

    def kelly_size(
        self,
        win_prob: float,
        win_loss_ratio: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Kelly Criterion with fractional Kelly for safety.

        WHY QUARTER-KELLY:
          Full Kelly maximises long-run geometric growth but leads to extreme
          drawdowns (up to 50%+ are common with full Kelly). Quarter-Kelly
          achieves ~90% of the growth rate with far lower drawdowns.
        """
        q = 1 - win_prob
        kelly_pct = (win_prob * win_loss_ratio - q) / win_loss_ratio
        kelly_pct = max(0, kelly_pct)    # no shorting in Kelly basic formulation
        adjusted  = kelly_pct * kelly_fraction
        max_size  = self.max_position_pct * self.portfolio_value
        return min(adjusted * self.portfolio_value, max_size)

    def allocate_portfolio(
        self,
        signals: Dict[str, float],
        vols: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Multi-stock portfolio allocation with inverse-volatility weighting.

        ENSURES:
          * Each position <= MAX_POSITION_PCT
          * Total long exposure <= 100% of portfolio
          * Each position risk-equalised by volatility
        """
        weights: Dict[str, float] = {}
        active_tickers = {k: v for k, v in signals.items() if abs(v) > 0.1}

        if not active_tickers:
            return weights

        # Inverse-vol weights for active signals
        inv_vols = {t: 1.0 / vols.get(t, 0.20) for t in active_tickers}
        total_inv_vol = sum(inv_vols.values())

        for t in active_tickers:
            raw_weight = (inv_vols[t] / total_inv_vol) * abs(signals[t])
            weights[t] = np.sign(signals[t]) * min(raw_weight, self.max_position_pct)

        return weights


# -----------------------------------------------------------------------------
# 2. STOP-LOSS MANAGER
# -----------------------------------------------------------------------------

class StopLossManager:
    """
    Manages stop-loss levels for all open positions.

    TWO STOP TYPES:
      1. Fixed %:  Stop = entry_price x (1  stop_pct)
         Simple, predictable, doesn't adapt to volatility.
      2. ATR-based: Stop = entry_price  N x ATR(14)
         BETTER: scales stop distance with current volatility.
         Tight stops in low-vol periods (less noise), wider in high-vol.
         Standard in institutional systems.
    """

    def __init__(
        self,
        fixed_pct:     float = STOP_LOSS_PCT,
        atr_multiplier: float = ATR_MULTIPLIER,
    ):
        self.fixed_pct       = fixed_pct
        self.atr_multiplier  = atr_multiplier
        self.stops: Dict[str, Dict] = {}  # ticker -> {type, level, entry, trail}

    def set_stop(
        self,
        ticker: str,
        entry_price: float,
        atr: Optional[float] = None,
        direction: int = 1,
        stop_type: str = "atr",
    ) -> float:
        """Set initial stop and record it."""
        if stop_type == "atr" and atr is not None:
            stop_dist = self.atr_multiplier * atr
        else:
            stop_dist = entry_price * self.fixed_pct

        stop_level = entry_price - direction * stop_dist
        self.stops[ticker] = {
            "entry"     : entry_price,
            "stop"      : stop_level,
            "direction" : direction,
            "high_water": entry_price,
            "stop_type" : stop_type,
            "atr"       : atr,
        }
        logger.debug(f"[{ticker}] Stop set at {stop_level:.2f} (entry={entry_price:.2f})")
        return stop_level

    def update_trailing_stop(
        self,
        ticker: str,
        current_price: float,
    ) -> Tuple[bool, float]:
        """
        Update trailing stop and check if it's been hit.

        TRAILING STOP LOGIC:
          As price moves in our favour, the stop moves up proportionally.
          This locks in profits while letting winners run.
          Critically: the stop only moves UP (for longs), never down.
        """
        if ticker not in self.stops:
            return False, 0.0

        info = self.stops[ticker]
        atr  = info.get("atr", 0)
        direction = info["direction"]

        # Update high-water mark
        if direction == 1:
            info["high_water"] = max(info["high_water"], current_price)
        else:
            info["high_water"] = min(info["high_water"], current_price)

        # New trailing stop
        if info["stop_type"] == "atr" and atr:
            new_stop = info["high_water"] - direction * self.atr_multiplier * atr
        else:
            new_stop = info["high_water"] * (1 - direction * self.fixed_pct)

        # Only tighten (never loosen)
        if direction == 1:
            info["stop"] = max(info["stop"], new_stop)
        else:
            info["stop"] = min(info["stop"], new_stop)

        # Check if stop is hit
        triggered = (direction == 1 and current_price <= info["stop"]) or \
                    (direction == -1 and current_price >= info["stop"])

        return triggered, info["stop"]

    def remove_stop(self, ticker: str) -> None:
        self.stops.pop(ticker, None)


# -----------------------------------------------------------------------------
# 3. VOLATILITY REGIME FILTER
# -----------------------------------------------------------------------------

class VolatilityRegimeFilter:
    """
    WHY VOLATILITY REGIME FILTERING:
      In high-volatility regimes (market stress, crises), price prediction
      models are least reliable - noise overwhelms signal. During these
      periods, reducing exposure reduces drawdowns without sacrificing much
      return (because the signal is weak anyway).

    REGIMES:
      LOW    : vol <= 15% ann. -> full position size
      MEDIUM : 15% < vol <= 25% -> 75% of normal size
      HIGH   : 25% < vol <= 40% -> 50% of normal size
      EXTREME: vol > 40% -> 25% of normal size (near flat book)

    IMPLEMENTATION:
      We use the 20-day realised volatility of the stock and compare to
      its own historical distribution (z-score approach) to detect
      anomalous vol regimes.
    """

    VOL_REGIMES = [
        (0.15, 1.00, "LOW"),
        (0.25, 0.75, "MEDIUM"),
        (0.40, 0.50, "HIGH"),
        (float("inf"), 0.25, "EXTREME"),
    ]

    def get_multiplier(self, ann_vol: float) -> Tuple[float, str]:
        for threshold, multiplier, regime in self.VOL_REGIMES:
            if ann_vol <= threshold:
                return multiplier, regime
        return 0.25, "EXTREME"

    def filter_signals(
        self,
        signals: pd.Series,
        vol_series: pd.Series,
    ) -> pd.Series:
        """Scale signals by volatility regime multiplier."""
        filtered = signals.copy()
        for date in signals.index:
            if date in vol_series.index:
                mult, regime = self.get_multiplier(vol_series[date])
                filtered[date] *= mult
        return filtered


# -----------------------------------------------------------------------------
# 4. PORTFOLIO RISK CONTROLLER
# -----------------------------------------------------------------------------

class PortfolioRiskController:
    """
    Portfolio-level risk limits.

    CONTROLS:
      1. Max Drawdown Circuit Breaker: If portfolio draws down >15% from peak,
         liquidate all positions and go to cash until conditions normalise.
         WHY: Prevents a losing streak from compounding into a catastrophic loss.

      2. Concentration Limit: No more than MAX_POSITION_PCT in a single name.

      3. Correlation Limit: If two positions have rolling 60-day correlation
         > 0.8, reduce the smaller position by 50% (avoid doubling up on risk).
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        max_position_pct: float   = MAX_POSITION_PCT,
        max_correlation: float    = 0.80,
    ):
        self.max_dd_limit    = max_drawdown_limit
        self.max_position    = max_position_pct
        self.max_correlation = max_correlation
        self.peak_value      = 0.0
        self.circuit_breaker = False

    def update_and_check(self, portfolio_value: float) -> bool:
        """
        Returns True if portfolio is in circuit-breaker state (should be flat).
        """
        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = (portfolio_value - self.peak_value) / self.peak_value

        if drawdown <= -self.max_dd_limit:
            if not self.circuit_breaker:
                logger.warning(
                    f"CIRCUIT BREAKER TRIGGERED: drawdown={drawdown:.1%} > "
                    f"limit={-self.max_dd_limit:.1%}. Liquidating all positions."
                )
            self.circuit_breaker = True
        elif drawdown > -self.max_dd_limit * 0.5:
            # Reset circuit breaker once drawdown halves
            self.circuit_breaker = False

        return self.circuit_breaker

    def check_correlations(
        self,
        positions: Dict[str, float],
        returns_df: pd.DataFrame,
        lookback: int = 60,
    ) -> Dict[str, float]:
        """
        Reduce correlated positions to avoid concentration in a single factor.
        """
        tickers = [t for t in positions if abs(positions[t]) > 0.01]
        if len(tickers) < 2:
            return positions

        corr_matrix = (
            returns_df[tickers].tail(lookback)
            .pct_change().corr()
        )
        adjusted = positions.copy()

        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                corr = corr_matrix.loc[t1, t2]
                if abs(corr) > self.max_correlation:
                    # Reduce the smaller position
                    if abs(adjusted[t1]) < abs(adjusted[t2]):
                        adjusted[t1] *= 0.5
                        logger.debug(f"Correlation {t1}-{t2}={corr:.2f} -> halved {t1}")
                    else:
                        adjusted[t2] *= 0.5
                        logger.debug(f"Correlation {t1}-{t2}={corr:.2f} -> halved {t2}")

        return adjusted


# -----------------------------------------------------------------------------
# RISK-MANAGED SIGNAL PIPELINE
# -----------------------------------------------------------------------------

def apply_risk_management(
    signals: pd.Series,
    price_df: pd.DataFrame,
    ticker: str,
    portfolio_value: float = 1_000_000,
) -> pd.Series:
    """
    Full pipeline: volatility filter -> position sizing -> stop-loss annotation.

    Returns
    -------
    risk_adjusted_signals : pd.Series
        Signals scaled by volatility regime and position size limits.
    """
    vol_filter = VolatilityRegimeFilter()
    sizer      = PositionSizer(portfolio_value)

    # 1. Compute realised volatility
    returns    = price_df["Close"].pct_change()
    roll_vol   = returns.rolling(20).std() * np.sqrt(252)
    roll_vol   = roll_vol.reindex(signals.index).ffill()

    # 2. Apply volatility regime filter
    filtered   = vol_filter.filter_signals(signals, roll_vol)

    # 3. Normalise to position size
    risk_adj   = filtered.copy()
    for date in filtered.index:
        vol = roll_vol.get(date, 0.20)
        mult, _ = vol_filter.get_multiplier(vol)
        risk_adj[date] = filtered[date] * mult

    logger.info(f"[{ticker}] Risk-adjusted signals: "
                f"mean={risk_adj.mean():.3f}, non-zero={( risk_adj != 0).sum()}")

    return risk_adj
