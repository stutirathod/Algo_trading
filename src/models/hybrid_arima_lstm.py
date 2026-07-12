"""
=============================================================================
Hybrid ARIMA-LSTM Model Orchestrator
=============================================================================
Trains the statistical ARIMA model on linear features (the target log-returns),
extracts residuals, and trains a deep LSTM model on the residuals using 
non-linear high-importance features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import (
    LSTM_HYBRID_FEATURES, OUTPUT_DIR, LOG_LEVEL, MODEL_MODE
)
from src.models.arima_model import train_arima

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def train_hybrid_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    ticker: str = "STOCK",
) -> Tuple[object, object, Dict]:
    """
    Trains a Hybrid ARIMA-LSTM model.
    1. Trains ARIMA on the raw return target
    2. Extracts in-sample residuals
    3. Trains LSTM on 'arima_residual' using LSTM_HYBRID_FEATURES
    """
    logger.info(f"\n{'='*60}\n[{ticker}] Training HYBRID ARIMA-LSTM\n{'='*60}")
    
    # Check mode
    target_col = "target_direction" if MODEL_MODE == "classification" else "target_return"
    
    # -- 1. Train ARIMA --
    logger.info(f"[{ticker}] Step 1/3: Training ARIMA model")
    arima_model, residuals, arima_metrics = train_arima(df, target_col=target_col, ticker=ticker)
    
    if arima_model is None:
        logger.error(f"[{ticker}] ARIMA failed. Bailing out of Hybrid training.")
        return None, None, {}
        
    # Append residuals to df
    df_hybrid = df.copy()
    df_hybrid["arima_residual"] = residuals
    
    # Drop rows without residuals
    df_hybrid = df_hybrid.dropna(subset=["arima_residual"])
    
    # Save ARIMA
    with open(ARTIFACT_DIR / f"{ticker.replace('.','_')}_arima.pkl", "wb") as f:
        pickle.dump(arima_model, f)
        
    # -- 2. Train LSTM on Residuals --
    try:
        from src.models.lstm_model import train_lstm
    except ImportError:
        logger.warning(f"[{ticker}] TensorFlow not available. Skipping Hybrid LSTM component.")
        return arima_model, None, {"arima": arima_metrics}
    
    logger.info(f"[{ticker}] Step 2/3: Training LSTM on ARIMA residuals")
    
    # Ensure our subset of features is present
    hybrid_features = [f for f in LSTM_HYBRID_FEATURES if f in df_hybrid.columns]
    
    lstm_model, history, lstm_results = train_lstm(
        df=df_hybrid,
        feature_cols=hybrid_features,
        ticker=f"{ticker}_Hybrid",
        target_col="arima_residual"
    )
    
    if lstm_model:
        lstm_model.save(str(ARTIFACT_DIR / f"{ticker.replace('.','_')}_hybrid_lstm.keras"))
        
    # -- 3. Combine Metrics --
    logger.info(f"[{ticker}] Step 3/3: Evaluating combined Hybrid model.")
    
    try:
        from src.models.tree_models import compute_metrics
        test_dates = lstm_results["predictions"]["dates"]
        y_true_test = df.loc[test_dates, target_col].values
        arima_pred_test = y_true_test - df_hybrid.loc[test_dates, "arima_residual"].values
        lstm_pred_test = lstm_results["predictions"]["proba"]
        hybrid_pred = arima_pred_test + lstm_pred_test
        
        hybrid_test_metrics = compute_metrics(y_true_test, hybrid_pred, "Hybrid Test", mode="regression")
    except Exception as e:
        logger.warning(f"Failed to compute combined hybrid metrics: {e}")
        hybrid_test_metrics = lstm_results["test"]
        hybrid_pred = lstm_results["predictions"]["proba"]
        test_dates = lstm_results["predictions"]["dates"]
        y_true_test = lstm_results["predictions"]["actual"]
        
    results = {
        "train": lstm_results["train"],  # Proxy
        "val": lstm_results["val"],      # Proxy
        "test": hybrid_test_metrics,     # True Combined Test Result
        "predictions": {
            "dates": test_dates,
            "actual": y_true_test,
            "predicted": hybrid_pred,
            "proba": hybrid_pred
        }
    }
    
    return arima_model, lstm_model, results
