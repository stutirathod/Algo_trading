"""
ARIMA Model for Hybrid Architecture
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    raise ImportError("statsmodels not installed. Run: pip install statsmodels")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import ARIMA_ORDER, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def train_arima(
    df: pd.DataFrame,
    target_col: str = "target_return",
    ticker: str = "STOCK",
    order: Tuple[int, int, int] = ARIMA_ORDER,
) -> Tuple[object, pd.Series, Dict]:
    """
    Trains an ARIMA model on the target variable.
    Returns:
      model: Trained statsmodels ARIMA model.
      residuals: In-sample residuals (Actual - Predicted) aligned with df.index.
      metrics: Dictionary of basic fit metrics.
    """
    logger.info(f"[{ticker}] Training ARIMA model (order={order}) on {target_col}...")
    
    y = df[target_col].dropna()
    
    try:
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        
        # Get in-sample predictions and align indices
        pred = model_fit.fittedvalues
        residuals = y - pred
        
        # Optional: basic metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "aic": model_fit.aic,
            "bic": model_fit.bic,
        }
        
        logger.info(f"[{ticker}] ARIMA Train | RMSE={rmse:.5f} | AIC={model_fit.aic:.1f}")
        
    except Exception as e:
        logger.error(f"[{ticker}] ARIMA training failed: {e}")
        model_fit = None
        residuals = pd.Series(0, index=y.index)
        metrics = {"rmse": 0.0, "mae": 0.0, "aic": 0.0, "bic": 0.0}
        
    return model_fit, residuals, metrics
