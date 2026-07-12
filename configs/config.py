"""
=============================================================================
ALGORITHMIC TRADING SYSTEM - CONFIGURATION
=============================================================================
Central configuration for all parameters. Separating config from code
ensures reproducibility and easy experimentation without touching model logic.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
CACHE_DIR   = DATA_DIR / "cache"
OUTPUT_DIR  = BASE_DIR / "outputs"
LOG_DIR     = BASE_DIR / "logs"

for d in [RAW_DIR, PROC_DIR, CACHE_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Universe -----------------------------------------------------------------
# NSE/BSE tickers. yfinance uses ".NS" suffix for NSE stocks.
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
]
BENCHMARK = "^NSEI"          # Nifty 50 index

# --- Data Collection ----------------------------------------------------------
DATA_START  = "2015-01-01"
DATA_END    = "2024-12-31"
INTERVAL    = "1d"           # Daily OHLCV

# --- Feature Engineering ------------------------------------------------------
# Moving average windows
MA_WINDOWS      = [5, 10, 20, 50, 200]
# Bollinger Band window & std multiplier
BB_WINDOW       = 20
BB_STD          = 2.0
# RSI window
RSI_WINDOW      = 14
# MACD parameters
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
# Volatility window (realised vol)
VOL_WINDOW      = 20
# Lag features to create (in days)
LAG_DAYS        = [1, 2, 3, 5, 10]
# Target: predict n-day forward return
TARGET_HORIZON  = 1          # 1 = next-day close prediction

# --- Model Mode ---------------------------------------------------------------
# "classification" trains models to predict Up/Down direction (recommended)
# "regression" trains models to predict continuous log-return
MODEL_MODE = "regression"

# --- Hybrid ARIMA-LSTM --------------------------------------------------------
ARIMA_ORDER = (5, 1, 0)
LSTM_HYBRID_FEATURES = [
    "rsi", "cci", "ema_5", "ema_20", "realised_vol", "volume_zscore",
    "macd", "bb_width", "atr_pct", "stoch_k", "williams_r", "roc_10", 
    "sma_5_20_cross", "price_to_vwap", "log_return"
]

# --- Train / Validation / Test Split ------------------------------------------
TRAIN_END   = "2021-12-31"
VAL_END     = "2022-12-31"
# Everything after VAL_END is out-of-sample test

# --- LSTM ---------------------------------------------------------------------
LSTM_LOOKBACK   = 30         # Reduced: daily market memory decays in ~2-4 weeks
LSTM_UNITS      = [32, 16]   # Reduced capacity - prevents memorising ~1500 rows
LSTM_DROPOUT    = 0.4        # Increased dropout - strong regularisation for small dataset
LSTM_EPOCHS     = 100
LSTM_BATCH      = 32
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 15         # early-stopping patience

# --- Random Forest ------------------------------------------------------------
RF_N_ESTIMATORS     = 500
RF_MAX_DEPTH        = None   # grow full trees; controlled by min_samples_leaf
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS           = -1
RF_RANDOM_STATE     = 42

# --- XGBoost ------------------------------------------------------------------
XGB_N_ESTIMATORS    = 1000
XGB_MAX_DEPTH       = 3       # Reduced: shallower trees generalise better on noisy returns
XGB_LEARNING_RATE   = 0.01
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE       = 0.5     # Reduced: force trees to use fewer correlated features
XGB_EARLY_STOPPING  = 50
XGB_RANDOM_STATE    = 42

# --- Ensemble -----------------------------------------------------------------
ENSEMBLE_METHOD = "stacking"   # "weighted" | "stacking" | "blending"
# Manual weights used only when ENSEMBLE_METHOD == "weighted"
ENSEMBLE_WEIGHTS = {"lstm": 0.4, "rf": 0.3, "xgb": 0.3}

# --- Backtesting --------------------------------------------------------------
INITIAL_CAPITAL     = 1_000_000   # INR
TRANSACTION_COST    = 0.001       # 0.1 % per trade (brokerage + STT approx)
SLIPPAGE            = 0.0005      # 0.05 % slippage
RISK_FREE_RATE      = 0.065       # ~6.5 % (RBI repo rate proxy)

# --- Risk Management ----------------------------------------------------------
STOP_LOSS_PCT       = 0.05        # 5 % trailing stop-loss
MAX_POSITION_PCT    = 0.10        # max 10 % of portfolio in one stock
VOLATILITY_SCALE    = True        # size positions inversely to realised vol
ATR_MULTIPLIER      = 2.0         # stop set at 2x ATR

# --- Logging ------------------------------------------------------------------
LOG_LEVEL = "INFO"
