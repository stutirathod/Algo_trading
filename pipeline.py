"""
=============================================================================
STEP 9 - SYSTEM ORCHESTRATION (MASTER PIPELINE)
=============================================================================
This script ties together all 8 steps into a single runnable pipeline.
It supports two modes:
  * TRAIN  : Full pipeline from data collection -> ensemble -> backtest
  * PREDICT: Load saved models -> generate tomorrow's signals

USAGE:
  python pipeline.py --mode train  --tickers RELIANCE.NS TCS.NS
  python pipeline.py --mode predict --tickers RELIANCE.NS
=============================================================================
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from configs.config import (
    TICKERS, ENSEMBLE_METHOD, OUTPUT_DIR, LOG_DIR,
    TRAIN_END, VAL_END, LSTM_LOOKBACK, LOG_LEVEL, MODEL_MODE
)
from src.data.collector   import collect_all_data
from src.data.eda         import run_eda
from src.features.engineer import build_features, preprocess, get_feature_summary
from src.models.tree_models  import (
    train_random_forest, train_xgboost, compare_models, chronological_split
)
from src.ensemble.combiner   import run_ensemble
from src.backtest.backtester import run_backtest
from src.risk.risk_manager   import apply_risk_management

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# HELPER: FEATURE COLUMN DETECTION
# -----------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return engineered feature columns (exclude raw OHLCV and targets)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Adj Close",
               "target_return", "target_direction"}
    return [c for c in df.columns if c not in exclude and not df[c].isnull().all()]


# -----------------------------------------------------------------------------
# STEP A: DATA + FEATURES
# -----------------------------------------------------------------------------

def step_data_and_features(
    tickers: List[str],
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Steps 1-3: Data collection -> EDA -> Feature engineering
    Returns dict[ticker -> processed feature DataFrame]
    """
    # -- Step 1: Collect ------------------------------------------------------
    price_data, fundamentals, ratios = collect_all_data(tickers, use_cache)

    # -- Step 3: Feature Engineering -----------------------------------------
    # (We run EDA after features so we can include engineered features in EDA)
    feature_store = build_features(price_data, ratios)

    # -- Step 2: EDA ----------------------------------------------------------
    eda_results = run_eda(price_data, feature_store)
    logger.info("EDA complete. Plots saved to outputs/eda/")

    # -- Preprocessing --------------------------------------------------------
    processed: Dict[str, pd.DataFrame] = {}
    for ticker, df in feature_store.items():
        df_proc, params = preprocess(df)  # mask derived internally after dropna
        # Save scaler params
        with open(ARTIFACT_DIR / f"{ticker.replace('.','_')}_scaler.pkl", "wb") as f:
            pickle.dump(params, f)
        processed[ticker] = df_proc
        logger.info(f"[{ticker}] Processed shape: {df_proc.shape}")

    return processed, price_data, eda_results


# -----------------------------------------------------------------------------
# STEP B: TRAIN INDIVIDUAL MODELS
# -----------------------------------------------------------------------------

def step_train_models(
    processed: Dict[str, pd.DataFrame],
    use_lstm: bool = False,
) -> Dict[str, Dict]:
    """
    Steps 4-5: Train each model individually, evaluate, compare.
    Returns model artifacts and results by ticker.
    """
    results_by_ticker = {}

    for ticker, df in processed.items():
        logger.info(f"\n{'='*60}\nTraining models for {ticker}\n{'='*60}")

        # Drop only rows where the target itself is NaN (last horizon rows).
        # Do NOT call df.dropna() on all columns - rolling z-score features have
        # leading NaNs that would wipe out the whole DataFrame.
        target_col = "target_direction" if MODEL_MODE == "classification" else "target_return"
        df_clean    = df.dropna(subset=[target_col])
        # Additionally drop rows where ALL feature cols are NaN (fully empty rows)
        feature_cols = get_feature_cols(df_clean)
        df_clean = df_clean.dropna(subset=feature_cols, how="all")
        logger.info(f"[{ticker}] Using {len(feature_cols)} features, {len(df_clean)} rows.")

        ticker_results = {}

        # -- RF ----------------------------------------------------------------
        logger.info(f"\n[{ticker}] -- RANDOM FOREST --")
        rf_model, rf_res = train_random_forest(df_clean, feature_cols, ticker)
        ticker_results["rf"] = {"model": rf_model, "results": rf_res}
        with open(ARTIFACT_DIR / f"{ticker.replace('.','_')}_rf.pkl", "wb") as f:
            pickle.dump(rf_model, f)

        # -- XGBoost -----------------------------------------------------------
        logger.info(f"\n[{ticker}] -- XGBOOST --")
        xgb_model, xgb_res = train_xgboost(df_clean, feature_cols, ticker)
        ticker_results["xgb"] = {"model": xgb_model, "results": xgb_res}
        xgb_model.save_model(str(ARTIFACT_DIR / f"{ticker.replace('.','_')}_xgb.json"))

        # -- LSTM (optional - requires TF) -------------------------------------
        lstm_model = None
        lstm_res   = None
        if use_lstm:
            try:
                from src.models.lstm_model import train_lstm
                logger.info(f"\n[{ticker}] -- LSTM (optional) --")
                lstm_model, _, lstm_res = train_lstm(df_clean, feature_cols, ticker)
                lstm_model.save(str(ARTIFACT_DIR / f"{ticker.replace('.','_')}_lstm.keras"))
                ticker_results["lstm"] = {"model": lstm_model, "results": lstm_res}
            except ImportError:
                logger.warning(f"[{ticker}] TensorFlow not available. Skipping LSTM.")

        # -- Hybrid ARIMA-LSTM -------------------------------------------------
        hybrid_res = None
        try:
            from src.models.hybrid_arima_lstm import train_hybrid_model
            logger.info(f"\n[{ticker}] -- HYBRID ARIMA-LSTM --")
            hybrid_arima, hybrid_lstm, hybrid_res = train_hybrid_model(df_clean, feature_cols, ticker)
            if hybrid_res:
                ticker_results["hybrid"] = {"arima": hybrid_arima, "lstm": hybrid_lstm, "results": hybrid_res}
        except Exception as e:
            logger.warning(f"[{ticker}] Hybrid ARIMA-LSTM failed: {e}")

        # -- Model Comparison --------------------------------------------------
        comparison = compare_models(rf_res, xgb_res, lstm_res, hybrid_res, ticker)
        ticker_results["comparison"] = comparison
        comparison.to_csv(OUTPUT_DIR / f"{ticker.replace('.','_')}_model_comparison.csv")

        logger.info(f"\n[{ticker}] INDIVIDUAL MODEL EVALUATION COMPLETE")
        logger.info(f"[{ticker}] Best model by DirAcc: "
                    f"{comparison['DirAcc'].idxmax()} "
                    f"({comparison['DirAcc'].max():.3f})")

        ticker_results["feature_cols"] = feature_cols
        results_by_ticker[ticker] = ticker_results

    return results_by_ticker


# -----------------------------------------------------------------------------
# STEP C: ENSEMBLE
# -----------------------------------------------------------------------------

def step_ensemble(
    processed: Dict[str, pd.DataFrame],
    model_results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Step 6: Build and evaluate ensemble models."""
    ensemble_results = {}

    for ticker, df in processed.items():
        tr = model_results.get(ticker, {})
        if not tr:
            continue

        df_clean     = df.dropna()
        feature_cols = tr["feature_cols"]
        rf_model     = tr["rf"]["model"]
        xgb_model    = tr["xgb"]["model"]
        lstm_model   = tr.get("lstm", {}).get("model")
        lstm_res     = tr.get("lstm", {}).get("results")
        rf_res       = tr["rf"]["results"]
        xgb_res      = tr["xgb"]["results"]

        y_final, metrics = run_ensemble(
            df=df_clean,
            feature_cols=feature_cols,
            rf_model=rf_model,
            xgb_model=xgb_model,
            rf_results=rf_res,
            xgb_results=xgb_res,
            lstm_model=lstm_model,
            lstm_results=lstm_res,
            method=ENSEMBLE_METHOD,
            ticker=ticker,
            lstm_lookback=LSTM_LOOKBACK,
        )

        ensemble_results[ticker] = {
            "predictions": y_final,
            "metrics": metrics,
        }

    return ensemble_results


# -----------------------------------------------------------------------------
# STEP D: BACKTEST
# -----------------------------------------------------------------------------

def step_backtest(
    price_data: Dict[str, pd.DataFrame],
    processed: Dict[str, pd.DataFrame],
    ensemble_results: Dict[str, Dict],
    model_results: Dict[str, Dict],
) -> None:
    """Steps 7-8: Run backtest with risk management on ensemble signals."""
    all_bt_metrics = []

    for ticker in ensemble_results:
        if ticker not in price_data:
            continue

        df_proc   = processed[ticker].dropna()
        test_df   = df_proc[df_proc.index > VAL_END]
        y_pred    = ensemble_results[ticker]["predictions"]
        pred_dates = test_df.index[-len(y_pred):]

        # Risk-managed signals
        raw_signals = pd.Series(y_pred, index=pred_dates)
        risk_signals = apply_risk_management(
            raw_signals, price_data[ticker].loc[pred_dates], ticker)

        equity_curve, metrics = run_backtest(
            price_df     = price_data[ticker],
            predictions  = risk_signals.values,
            pred_dates   = pred_dates,
            ticker       = ticker,
            strategy_name = "Ensemble+RiskMgmt",
        )

        metrics["ticker"] = ticker
        all_bt_metrics.append(metrics)

    # Save combined summary
    if all_bt_metrics:
        summary = pd.DataFrame(all_bt_metrics).set_index("ticker")
        summary.to_csv(OUTPUT_DIR / "backtest_summary.csv")
        logger.info(f"\n{'='*60}\nFINAL BACKTEST SUMMARY\n{'='*60}\n{summary.to_string()}")


# -----------------------------------------------------------------------------
# PREDICT MODE: GENERATE TODAY'S SIGNALS
# -----------------------------------------------------------------------------

def predict_today(tickers: List[str]) -> pd.DataFrame:
    """
    Load saved models + latest data -> generate next-day signals.

    WORKFLOW:
      1. Fetch latest price data (last 300 days for feature computation)
      2. Build features on latest data
      3. Load saved RF + XGB models
      4. Generate predictions
      5. Apply risk management
      6. Output ranked signal table
    """
    logger.info("PREDICT MODE - Generating tomorrow's signals")
    from src.data.collector import fetch_price_data, fetch_fundamental_data, compute_fundamental_ratios
    import xgboost as xgb_lib
    from datetime import date, timedelta

    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    price_data   = fetch_price_data(tickers, start=start, end=end, use_cache=False)
    fundamentals = fetch_fundamental_data(tickers)
    ratios       = compute_fundamental_ratios(fundamentals, price_data)

    from src.features.engineer import build_features
    feature_store = build_features(price_data, ratios)

    signals_out = []
    for ticker in tickers:
        if ticker not in feature_store:
            continue

        df          = feature_store[ticker].dropna()
        feature_cols = get_feature_cols(df)

        # Load scaler
        scaler_path = ARTIFACT_DIR / f"{ticker.replace('.','_')}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            df[feature_cols] = (df[feature_cols].clip(
                lower=scaler["lower"], upper=scaler["upper"], axis=1) -
                scaler["mean"]) / scaler["std"]

        latest = df[feature_cols].tail(1).values

        # Predict with available models
        preds = {}
        rf_path = ARTIFACT_DIR / f"{ticker.replace('.','_')}_rf.pkl"
        if rf_path.exists():
            with open(rf_path, "rb") as f:
                rf = pickle.load(f)
            if MODEL_MODE == "classification":
                preds["rf"] = float(rf.predict_proba(latest)[0, 1])  # P(Up)
            else:
                preds["rf"] = float(rf.predict(latest)[0])

        xgb_path = ARTIFACT_DIR / f"{ticker.replace('.','_')}_xgb.json"
        if xgb_path.exists():
            xgb_m = xgb_lib.XGBClassifier() if MODEL_MODE == "classification" else xgb_lib.XGBRegressor()
            xgb_m.load_model(str(xgb_path))
            if MODEL_MODE == "classification":
                preds["xgb"] = float(xgb_m.predict_proba(latest)[0, 1])  # P(Up)
            else:
                preds["xgb"] = float(xgb_m.predict(latest)[0])

        if preds:
            if MODEL_MODE == "classification":
                # Average P(Up) acros models, convert to [-1, 1] signal
                avg_prob       = np.mean(list(preds.values()))
                ensemble_pred  = 2 * avg_prob - 1
                conviction     = abs(ensemble_pred)
            else:
                ensemble_pred = np.mean(list(preds.values()))
                conviction    = abs(ensemble_pred)
            current_price = float(price_data[ticker]["Close"].iloc[-1])
            signals_out.append({
                "ticker"        : ticker,
                "signal"        : round(ensemble_pred, 6),
                "direction"     : "BUY" if ensemble_pred > 0 else "SELL",
                "conviction"    : round(conviction, 6),
                "current_price" : current_price,
                "model_signals" : preds,
            })

    df_signals = pd.DataFrame(signals_out).sort_values("conviction", ascending=False)
    logger.info(f"\nSignals for {date.today()}:\n{df_signals.to_string()}")
    df_signals.to_csv(OUTPUT_DIR / f"signals_{date.today()}.csv", index=False)
    return df_signals


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Algorithmic Trading System")
    parser.add_argument("--mode",    choices=["train", "predict"], default="train")
    parser.add_argument("--tickers", nargs="+", default=TICKERS[:3])  # default: first 3
    parser.add_argument("--lstm",    action="store_true", help="Include LSTM training")
    parser.add_argument("--cache",   action="store_true", default=True,
                        help="Use cached data")
    args = parser.parse_args()

    if args.mode == "train":
        logger.info("=" * 70)
        logger.info("ALGORITHMIC TRADING SYSTEM - FULL TRAINING PIPELINE")
        logger.info("=" * 70)

        # Steps 1-3
        processed, price_data, eda = step_data_and_features(args.tickers, args.cache)

        # Steps 4-5
        model_results = step_train_models(processed, use_lstm=args.lstm)

        # Step 6
        ensemble_results = step_ensemble(processed, model_results)

        # Steps 7-8
        step_backtest(price_data, processed, ensemble_results, model_results)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"All outputs saved to: {OUTPUT_DIR}")
        logger.info("=" * 70)

    elif args.mode == "predict":
        signals = predict_today(args.tickers)
        print(signals.to_string())


if __name__ == "__main__":
    main()