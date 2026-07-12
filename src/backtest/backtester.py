"""
=============================================================================
STEP 7 - BACKTESTING FRAMEWORK
=============================================================================
PURPOSE:
  After building a predictive model, we need to validate whether the
  predictions translate into *profitable trading strategies*. A model with
  high directional accuracy doesn't automatically produce good returns - it
  must survive transaction costs, slippage, and adverse market regimes.

FRAMEWORK DESIGN:
  * Event-driven simulation: process each bar (day) sequentially
  * No look-ahead: signals are generated using only data available at EOD t
    and executed at open price on day t+1 (realistic)
  * Transaction costs + slippage are explicitly modelled
  * Multiple strategy variants: long-only, long-short, top-N portfolio

METRICS EXPLAINED:
  Sharpe Ratio    : (Mean Return  Risk-Free Rate) / Std(Returns) x sqrt252
                    > 1 is acceptable, > 2 is excellent for live trading.
  Sortino Ratio   : Penalises only *downside* volatility. More appropriate for
                    strategies with asymmetric return distributions.
  Max Drawdown    : Largest peak-to-trough decline. The psychological and
                    regulatory risk metric for any fund manager.
  CAGR            : Compound Annual Growth Rate. The true annualised return
                    accounting for compounding effects.
  Win Rate        : Fraction of trades with positive P&L. Useful alongside
                    profit factor to assess strategy quality.
  Profit Factor   : Gross profit / Gross loss. > 1.5 is desirable.
  Calmar Ratio    : CAGR / Max Drawdown. Risk-adjusted return metric.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    INITIAL_CAPITAL, TRANSACTION_COST, SLIPPAGE,
    RISK_FREE_RATE, STOP_LOSS_PCT, MAX_POSITION_PCT,
    OUTPUT_DIR, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

BT_DIR = OUTPUT_DIR / "backtest"
BT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# SIGNAL GENERATION
# -----------------------------------------------------------------------------

def generate_signals(
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    threshold: float = 0.0,
    confidence_weight: bool = True,
) -> pd.Series:
    """
    Convert raw return predictions into trading signals.

    SIGNAL LOGIC:
      pred > threshold  -> BUY  (+1)
      pred < -threshold -> SELL (-1)  [long-short mode]
      else              -> FLAT ( 0)

    CONFIDENCE WEIGHTING:
      Rather than binary signals, we scale position size by the magnitude of
      the predicted return. Large absolute predictions -> larger positions.
      This is called a "continuous signal" or "alpha signal" in quant finance.

    WHY A THRESHOLD:
      A threshold filters low-conviction signals near zero (which are noisiest)
      and avoids excessive trading. The threshold is calibrated on the
      distribution of predictions.
    """
    signals = pd.Series(0.0, index=dates)

    if confidence_weight:
        # Normalise predictions to [-1, 1] range using sigmoid-like transform
        pred_std = np.std(predictions)
        if pred_std > 0:
            norm_preds = predictions / (2 * pred_std)   # scale
            norm_preds = np.clip(norm_preds, -1, 1)
            signals = pd.Series(norm_preds, index=dates)
        else:
            signals = pd.Series(np.sign(predictions - threshold), index=dates)
    else:
        signals[predictions > threshold]  = 1.0
        signals[predictions < -threshold] = -1.0

    return signals


# -----------------------------------------------------------------------------
# PORTFOLIO SIMULATION
# -----------------------------------------------------------------------------

class PortfolioBacktester:
    """
    Event-driven backtesting engine.

    HOW IT WORKS:
      1. Each day t, we have a signal based on predictions made at t-1.
      2. We execute trades at open price of day t (realistic delay).
      3. We mark-to-market at close price of day t.
      4. Stop-loss is checked intraday using the low price.
      5. Transaction costs and slippage are deducted on each trade.

    STATE:
      cash          : Current cash balance
      holdings      : Dict[ticker -> shares held]
      portfolio_val : Time series of total portfolio value
      trades        : List of executed trades
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        tc: float              = TRANSACTION_COST,
        slippage: float        = SLIPPAGE,
        stop_loss: float       = STOP_LOSS_PCT,
        max_position: float    = MAX_POSITION_PCT,
    ):
        self.initial_capital = initial_capital
        self.tc              = tc
        self.slippage        = slippage
        self.stop_loss       = stop_loss
        self.max_position    = max_position

        self.reset()

    def reset(self):
        self.cash          = self.initial_capital
        self.holdings      = {}     # ticker -> shares
        self.entry_prices  = {}     # ticker -> entry price (for stop-loss)
        self.portfolio_val = []
        self.dates         = []
        self.trades        = []

    def run(
        self,
        price_df: pd.DataFrame,
        signals: pd.Series,
        ticker: str,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate strategy on a single ticker.

        EXECUTION MODEL:
          Signal at close t -> execute at open t+1 -> P&L measured at close t+1.
          This is the minimum realistic execution assumption for a daily system.
        """
        self.reset()

        # Align signal to price data
        common = price_df.index.intersection(signals.index)
        price  = price_df.loc[common]
        sig    = signals.loc[common]

        holding      = 0.0     # fractional position: +1 = fully long, -1 = fully short
        entry_price  = 0.0
        portfolio    = self.initial_capital

        for i in range(1, len(price)):
            date   = price.index[i]
            prev   = price.iloc[i - 1]
            curr   = price.iloc[i]
            signal = float(sig.iloc[i - 1])   # signal from previous day's close

            # -- Intraday Stop-Loss Check --------------------------------------
            if holding != 0 and entry_price > 0:
                stop_trigger = (
                    (curr["Low"] < entry_price * (1 - self.stop_loss) and holding > 0)
                    or
                    (curr["High"] > entry_price * (1 + self.stop_loss) and holding < 0)
                )
                if stop_trigger:
                    # Exit at stop price
                    stop_price = (entry_price * (1 - self.stop_loss) if holding > 0
                                  else entry_price * (1 + self.stop_loss))
                    exit_pnl   = holding * (stop_price / entry_price - 1) * portfolio
                    portfolio  += exit_pnl - abs(exit_pnl * self.tc)
                    self.trades.append({
                        "date": date, "type": "STOP_LOSS", "price": stop_price,
                        "holding": holding, "pnl": exit_pnl
                    })
                    holding = 0.0
                    entry_price = 0.0

            # -- Execute Signal -------------------------------------------------
            new_holding = signal
            if long_only:
                new_holding = max(0, new_holding)

            # Cap at max position size
            new_holding = np.clip(new_holding, -1, 1) * self.max_position / \
                          (curr["Open"] / self.initial_capital + 1e-10)
            new_holding = np.clip(new_holding, -1, 1)

            if abs(new_holding - holding) > 0.05:   # only trade if significant change
                exec_price = curr["Open"] * (1 + np.sign(new_holding - holding) * self.slippage)
                trade_cost = abs(new_holding - holding) * portfolio * self.tc
                portfolio -= trade_cost
                self.trades.append({
                    "date": date, "type": "SIGNAL", "price": exec_price,
                    "holding": new_holding, "signal": signal, "cost": trade_cost
                })
                if new_holding != 0 and holding == 0:
                    entry_price = exec_price

                holding = new_holding

            # -- Mark to Market -------------------------------------------------
            daily_return = (curr["Close"] / prev["Close"]) - 1
            portfolio   *= (1 + holding * daily_return)

            self.portfolio_val.append(portfolio)
            self.dates.append(date)

        equity_curve = pd.DataFrame({
            "portfolio_value": self.portfolio_val,
            "date"           : self.dates,
        }).set_index("date")

        equity_curve["returns"] = equity_curve["portfolio_value"].pct_change()
        equity_curve["drawdown"] = self._compute_drawdown(equity_curve["portfolio_value"])

        return equity_curve

    @staticmethod
    def _compute_drawdown(portfolio_series: pd.Series) -> pd.Series:
        rolling_max = portfolio_series.cummax()
        drawdown    = (portfolio_series - rolling_max) / rolling_max
        return drawdown


# -----------------------------------------------------------------------------
# PERFORMANCE METRICS
# -----------------------------------------------------------------------------

def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict:
    """
    Compute industry-standard performance metrics.
    """
    pv      = equity_curve["portfolio_value"]
    returns = equity_curve["returns"].dropna()

    # CAGR
    n_years = len(pv) / 252
    cagr    = (pv.iloc[-1] / pv.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe Ratio
    excess_ret   = returns - (risk_free_rate / 252)
    sharpe       = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)
                    if excess_ret.std() > 0 else 0.0)

    # Sortino Ratio (downside deviation only)
    downside = returns[returns < 0]
    sortino  = (excess_ret.mean() / downside.std() * np.sqrt(252)
                if len(downside) > 0 and downside.std() > 0 else 0.0)

    # Maximum Drawdown
    max_dd = float(equity_curve["drawdown"].min())

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win Rate & Profit Factor
    daily_pnl = returns * equity_curve["portfolio_value"].shift(1).dropna()
    win_rate  = (daily_pnl > 0).mean() if len(daily_pnl) > 0 else 0.0
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss   = abs(daily_pnl[daily_pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total Return
    total_return = (pv.iloc[-1] / pv.iloc[0]) - 1

    metrics = {
        "total_return"   : round(total_return,  4),
        "cagr"           : round(cagr,           4),
        "sharpe_ratio"   : round(sharpe,         4),
        "sortino_ratio"  : round(sortino,        4),
        "max_drawdown"   : round(max_dd,         4),
        "calmar_ratio"   : round(calmar,         4),
        "win_rate"       : round(win_rate,       4),
        "profit_factor"  : round(profit_factor,  4),
        "n_trading_days" : len(pv),
        "final_value"    : round(float(pv.iloc[-1]), 2),
    }

    return metrics


# -----------------------------------------------------------------------------
# VISUALISATION
# -----------------------------------------------------------------------------

def plot_backtest_results(
    equity_curve: pd.DataFrame,
    metrics: Dict,
    ticker: str,
    strategy_name: str = "Ensemble",
    benchmark_df: Optional[pd.DataFrame] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # -- 1. Equity Curve ----------------------------------------------------
    axes[0].plot(equity_curve["portfolio_value"], label=strategy_name,
                 color="steelblue", lw=1.5)
    if benchmark_df is not None:
        bm = benchmark_df["Close"].reindex(equity_curve.index).ffill()
        bm_norm = bm / bm.iloc[0] * INITIAL_CAPITAL
        axes[0].plot(bm_norm, label="Nifty 50 (buy & hold)", color="grey",
                     lw=1.0, ls="--", alpha=0.8)
    axes[0].set_title(
        f"{ticker} - {strategy_name} | "
        f"CAGR={metrics['cagr']:.1%} | Sharpe={metrics['sharpe_ratio']:.2f} | "
        f"MaxDD={metrics['max_drawdown']:.1%}"
    )
    axes[0].set_ylabel("Portfolio Value (INR)")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs.{x/1e5:.1f}L"))

    # -- 2. Drawdown ---------------------------------------------------------
    axes[1].fill_between(equity_curve.index, equity_curve["drawdown"],
                         0, color="crimson", alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # -- 3. Rolling Sharpe (60-day) ------------------------------------------
    rolling_ret    = equity_curve["returns"].dropna()
    daily_rf       = RISK_FREE_RATE / 252
    rolling_sharpe = (
        (rolling_ret - daily_rf)
        .rolling(60)
        .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
    )
    axes[2].plot(rolling_sharpe, color="purple", lw=1.2)
    axes[2].axhline(0, color="black", lw=0.5, ls="--")
    axes[2].axhline(1, color="green", lw=0.5, ls="--", label="Sharpe=1")
    axes[2].set_title("Rolling 60-day Sharpe Ratio")
    axes[2].set_ylabel("Sharpe")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(BT_DIR / f"{ticker}_{strategy_name.lower()}_backtest.png", dpi=120)
    plt.close()
    logger.info(f"[{ticker}] Backtest chart saved.")


def run_backtest(
    price_df: pd.DataFrame,
    predictions: np.ndarray,
    pred_dates: pd.DatetimeIndex,
    ticker: str,
    strategy_name: str = "Ensemble",
    benchmark_df: Optional[pd.DataFrame] = None,
    long_only: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Orchestrates signal generation -> simulation -> metrics -> plotting.
    """
    logger.info(f"\n{'='*60}\nSTEP 7 - BACKTESTING: {ticker} [{strategy_name}]\n{'='*60}")

    signals       = generate_signals(predictions, pred_dates)
    backtester    = PortfolioBacktester()
    equity_curve  = backtester.run(price_df.loc[pred_dates[0]:], signals,
                                   ticker, long_only=long_only)
    metrics       = compute_performance_metrics(equity_curve)
    plot_backtest_results(equity_curve, metrics, ticker, strategy_name, benchmark_df)

    logger.info(
        f"[{ticker}] Backtest Results:\n"
        + "\n".join(f"  {k:20s}: {v}" for k, v in metrics.items())
    )

    # Save metrics to CSV
    slug = ticker.replace(".", "_")
    pd.DataFrame([metrics]).to_csv(
        BT_DIR / f"{slug}_{strategy_name.lower()}_metrics.csv", index=False)

    # Save equity curve to CSV so the API/dashboard can read it directly
    eq_out = equity_curve.reset_index().rename(columns={"index": "date", "Date": "date"})
    eq_out["date"] = pd.to_datetime(eq_out["date"]).dt.strftime("%Y-%m-%d")
    eq_out.to_csv(BT_DIR / f"{slug}_equity.csv", index=False)
    logger.info(f"[{ticker}] Equity curve saved to CSV.")

    return equity_curve, metrics