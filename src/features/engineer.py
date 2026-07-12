"""
=============================================================================
STEP 3 - FEATURE ENGINEERING
=============================================================================
Three categories of features are created:
  A) Market / OHLCV features
  B) Technical indicators
  C) Fundamental ratios (aligned to daily dates)

DESIGN PRINCIPLE - NO LOOK-AHEAD BIAS:
  Every feature at time t is computed using only data available at t-1 or
  earlier. This is enforced by:
    * Using .shift(1) before computing any rolling statistic used as a feature
    * Forward-filling fundamentals from announcement date only

WHY EACH FEATURE CATEGORY:
  Market:       Raw price-action context. Returns capture momentum/mean-rev.
  Technical:    Rule-based signals used by professional traders; encode
                market psychology and supply/demand dynamics.
  Fundamental:  Business quality features that drive *long-term* trends;
                prevent overfitting to short-term noise.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta  # 'ta' library (pip install ta) for technical indicators

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    MA_WINDOWS, BB_WINDOW, BB_STD, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    VOL_WINDOW, LAG_DAYS, TARGET_HORIZON,
    PROC_DIR, LOG_LEVEL, MODEL_MODE
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# A. MARKET FEATURES
# -----------------------------------------------------------------------------

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY EACH FEATURE:
      log_return   : Log-returns are additive over time and more normally
                     distributed than simple returns -> better behaved for ML.
      range_pct    : (High-Low)/Close measures intraday uncertainty / volatility.
      gap_pct      : Open vs previous Close; captures overnight news impact.
      close_pos    : Where in the day's range the close fell (0=low, 1=high);
                     encodes buying/selling pressure.
      realised_vol : Rolling standard deviation of returns; the single most
                     important risk feature.
      volume_zscore: Abnormal volume often precedes large price moves.
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    df["log_return"]    = np.log(c / c.shift(1))
    df["return_1d"]     = c.pct_change(1)
    df["return_5d"]     = c.pct_change(5)
    df["return_10d"]    = c.pct_change(10)
    df["return_20d"]    = c.pct_change(20)

    df["range_pct"]     = (h - l) / c.shift(1)
    df["gap_pct"]       = (o - c.shift(1)) / c.shift(1)
    df["close_pos"]     = (c - l) / (h - l).replace(0, np.nan)

    df["realised_vol"]  = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252)

    vol_mean = v.rolling(20).mean()
    vol_std  = v.rolling(20).std()
    df["volume_zscore"] = (v - vol_mean) / vol_std.replace(0, np.nan)
    df["volume_ratio"]  = v / vol_mean   # today vs 20d avg

    # Lag returns
    for lag in LAG_DAYS:
        df[f"return_lag{lag}"] = df["log_return"].shift(lag)

    return df


# -----------------------------------------------------------------------------
# B. TECHNICAL INDICATORS
# -----------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the `ta` library which implements indicators without lookahead bias
    when shift is applied before use.

    WHY EACH GROUP:
      Moving averages:  Trend direction & strength. Multiple windows capture
                        short/medium/long-term momentum.
      RSI:              Overbought/oversold oscillator. Mean-reversion signal.
      MACD:             Momentum cross-over signal used by institutional traders.
      Bollinger Bands:  Volatility-normalised oscillator; position within bands
                        is a strong short-term signal.
      VWAP:             Institutional benchmark price; deviation from VWAP
                        identifies discount/premium.
      OBV:              Cumulative volume tells whether volume supports the trend.
      ATR:              Absolute measure of volatility; used for stop-loss sizing.
      Stochastic:       Similar to RSI; %K/%D crossover is a trading signal.
      Williams %R:      Momentum oscillator; good for identifying reversals.
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # -- Moving Averages - keep only the most informative windows --------------
    # We use a reduced set (5, 20, 200) to avoid near-duplicate features;
    # SMA_5 vs SMA_10 carry almost identical information and inflate dimensionality.
    essential_windows = [5, 20, 200]
    for w in essential_windows:
        ma = c.rolling(w).mean()
        df[f"sma_{w}"]         = ma
        df[f"price_to_sma{w}"] = (c / ma) - 1    # normalised distance from MA

    # EMA for short and long term only
    df["ema_5"]  = c.ewm(span=5,   adjust=False).mean()
    df["ema_20"] = c.ewm(span=20,  adjust=False).mean()

    # Essential MA cross signals
    df["sma_5_20_cross"]   = np.sign(df["sma_5"]  - df["sma_20"])
    df["sma_20_200_cross"] = np.sign(df["sma_20"] - df["sma_200"])  # Golden/Death cross

    # -- RSI -------------------------------------------------------------------
    rsi = ta.momentum.RSIIndicator(close=c, window=RSI_WINDOW)
    df["rsi"] = rsi.rsi()
    df["rsi_zscore"] = (df["rsi"] - df["rsi"].rolling(252).mean()) / \
                        df["rsi"].rolling(252).std()

    # -- MACD ------------------------------------------------------------------
    macd_ind = ta.trend.MACD(close=c, window_fast=MACD_FAST,
                              window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"]   = macd_ind.macd_diff()   # histogram; positive = bullish momentum

    # -- Bollinger Bands -------------------------------------------------------
    bb = ta.volatility.BollingerBands(close=c, window=BB_WINDOW, window_dev=BB_STD)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]   # normalised width
    df["bb_pct"]    = bb.bollinger_pband()   # 0-1 position within bands

    # -- VWAP ------------------------------------------------------------------
    # True daily VWAP requires tick data; we approximate with (H+L+C)/3 x Volume
    # cumulated over a rolling 20-day window as an institutional price proxy.
    typical = (h + l + c) / 3
    df["vwap_20"]          = (typical * v).rolling(20).sum() / v.rolling(20).sum()
    df["price_to_vwap"]    = (c / df["vwap_20"]) - 1

    # -- On-Balance Volume -----------------------------------------------------
    obv = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v)
    df["obv"]           = obv.on_balance_volume()
    df["obv_ma20"]      = df["obv"].rolling(20).mean()
    df["obv_signal"]    = np.sign(df["obv"] - df["obv_ma20"])   # OBV vs its MA

    # -- ATR (Average True Range) ----------------------------------------------
    atr = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=14)
    df["atr"]           = atr.average_true_range()
    df["atr_pct"]       = df["atr"] / c      # normalise by price for cross-stock use

    # -- Stochastic ------------------------------------------------------------
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = np.sign(df["stoch_k"] - df["stoch_d"])

    # -- Williams %R -----------------------------------------------------------
    wr = ta.momentum.WilliamsRIndicator(high=h, low=l, close=c, lbp=14)
    df["williams_r"] = wr.williams_r()

    # -- Momentum / Rate of Change ---------------------------------------------
    for w in [10, 20, 50]:
        roc = ta.momentum.ROCIndicator(close=c, window=w)
        df[f"roc_{w}"] = roc.roc()

    # -- Commodity Channel Index -----------------------------------------------
    cci = ta.trend.CCIIndicator(high=h, low=l, close=c, window=20)
    df["cci"] = cci.cci()

    return df


# -----------------------------------------------------------------------------
# C. FUNDAMENTAL FEATURES
# -----------------------------------------------------------------------------

def add_fundamental_features(
    df: pd.DataFrame,
    ratio_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    WHY FUNDAMENTALS IN A PRICE PREDICTION MODEL:
      - EPS growth & ROE drive analyst upgrades -> systematic price moves.
      - High D/E ratio means leverage risk -> greater downside in market stress.
      - FCF yield is a value signal used by institutional investors.
      - These features are especially useful for the *long-term trend* model.

    The ratio_df is aligned to price dates via forward-fill (no lookahead).
    """
    if ratio_df is None or ratio_df.empty:
        return df

    # Align to the price DataFrame's index
    ratio_aligned = ratio_df.reindex(df.index).ffill()
    for col in ratio_aligned.columns:
        df[f"fund_{col}"] = ratio_aligned[col]

    return df


# -----------------------------------------------------------------------------
# D. TARGET VARIABLE CREATION
# -----------------------------------------------------------------------------

def create_targets(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Two targets are created:
      1. target_return: Forward n-day log-return (regression)
         WHY LOG-RETURN: Symmetric around 0; additive; numerically stable.
      2. target_direction: 1 if price goes up, 0 if down (classification)
         WHY DIRECTION: Directional accuracy is often more useful than raw
         price prediction for generating trading signals.
    """
    c = df["Close"]
    df["target_return"]    = np.log(c.shift(-horizon) / c)   # future return
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


# -----------------------------------------------------------------------------
# PREPROCESSING: SCALING & CLEANING
# -----------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    train_mask=None,      # kept for backward-compat but ignored; mask recomputed internally
    winsorise_pct: float = 0.01,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Steps:
      1. Drop rows with NaN in target (last `horizon` rows by construction)
      2. Winsorise features at 1st/99th percentile (handles outlier inputs)
      3. Standardise (z-score) using ONLY train-set statistics
         WHY: Prevents data leakage from val/test into scaling parameters.

    NOTE: train_mask is now derived INTERNALLY after dropna() to avoid the
    IndexError caused by length mismatch when an external boolean mask was
    computed on the pre-dropna DataFrame.

    Returns scaled DataFrame and scaler params dict.
    """
    from configs.config import TRAIN_END
    # Drop target-NaN rows FIRST, then recompute mask from the surviving index
    df = df.dropna(subset=["target_return"])
    train_mask = df.index <= TRAIN_END

    # Feature columns (everything except raw OHLCV and targets)
    raw_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close",
                "target_return", "target_direction"]
    feature_cols = [c for c in df.columns if c not in raw_cols]

    # Winsorise using 1st/99th percentile of train set
    train_data = df.loc[train_mask, feature_cols]
    lower = train_data.quantile(winsorise_pct)
    upper = train_data.quantile(1 - winsorise_pct)
    df[feature_cols] = df[feature_cols].clip(lower=lower, upper=upper, axis=1)

    # -- Rolling Z-score Standardisation --------------------------------------
    # WHY ROLLING Z-SCORE OVER STATIC Z-SCORE:
    #   Financial indicators drift with market regimes. A "high" RSI (70+) in a
    #   2018 bear market is normal in a 2021 bull market. Static mean/std treats
    #   all regimes equally. Rolling Z-scores are regime-invariant.
    #
    # ADAPTIVE WINDOW: If the dataset is small (e.g. cache only has 72 rows),
    # we shrink the window so at least 50% of rows get rolling normalization.
    n_rows = len(df)
    ROLLING_WINDOW = min(252, max(20, n_rows // 4))   # adaptive, at least 20
    min_periods    = min(60,  max(10, n_rows // 8))   # adaptive min_periods

    mu  = df[feature_cols].rolling(ROLLING_WINDOW, min_periods=min_periods).mean()
    std = df[feature_cols].rolling(ROLLING_WINDOW, min_periods=min_periods).std()
    std = std.replace(0, np.nan).fillna(1)   # avoid div-by-zero; NaN->1
    df_scaled = df.copy()
    df_scaled[feature_cols] = (df[feature_cols] - mu) / std

    # Fall back to train-set static z-score for early rows where rolling window hasn't filled
    static_mu  = train_data.mean()
    static_std = train_data.std().replace(0, 1)
    static_scaled = (df[feature_cols] - static_mu) / static_std
    # Use rolling wherever rolling is available (non-NaN), else static
    mask_use_rolling = mu.notna().all(axis=1)
    df_scaled.loc[~mask_use_rolling, feature_cols] = static_scaled.loc[~mask_use_rolling]
    df = df_scaled

    scaler_params = {"mean": static_mu, "std": static_std, "lower": lower, "upper": upper,
                     "feature_cols": feature_cols}

    return df, scaler_params


# -----------------------------------------------------------------------------
# MASTER FEATURE ENGINEERING FUNCTION
# -----------------------------------------------------------------------------

def build_features(
    price_data: Dict[str, pd.DataFrame],
    ratios: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Applies the full feature engineering pipeline to every ticker.
    Saves processed feature DataFrames to disk.
    """
    logger.info("=" * 60)
    logger.info("STEP 3 - FEATURE ENGINEERING")
    logger.info("=" * 60)

    feature_store: Dict[str, pd.DataFrame] = {}

    for ticker, price_df in price_data.items():
        logger.info(f"[{ticker}] Building features...")
        df = price_df.copy()

        # Market features
        df = add_market_features(df)

        # Technical indicators
        df = add_technical_indicators(df)

        # Fundamental ratios
        ratio_df = ratios.get(ticker) if ratios else None
        df = add_fundamental_features(df, ratio_df)

        # Target
        df = create_targets(df)

        # Drop early rows where rolling windows haven't filled (MA-200 needs 200 days)
        df.dropna(subset=["sma_200", "log_return"], inplace=True)

        # Report
        n_features = len([c for c in df.columns
                          if c not in ["Open","High","Low","Close","Volume",
                                       "Adj Close","target_return","target_direction"]])
        logger.info(f"[{ticker}] Total features: {n_features} | Rows: {len(df)}")

        # Save
        out_path = PROC_DIR / f"{ticker.replace('.','_')}_features.parquet"
        df.to_parquet(out_path)

        feature_store[ticker] = df

    logger.info("STEP 3 - FEATURE ENGINEERING COMPLETE\n")
    return feature_store


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of all features: count, mean, std, nulls."""
    raw_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    feat_df  = df.drop(columns=raw_cols, errors="ignore")
    summary  = feat_df.describe().T
    summary["null_pct"] = feat_df.isnull().mean() * 100
    return summary.round(4)