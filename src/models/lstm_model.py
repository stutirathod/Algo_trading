"""
=============================================================================
STEP 4 & 5 - MODEL 1: LSTM (Long Short-Term Memory) - Classification Mode
=============================================================================
WHY LSTM FOR THIS PROBLEM:
  Stock price direction prediction benefits from sequential context.
  Yesterday's momentum, recent RSI trend, and volume pattern over the past
  30 days all contribute to whether tomorrow's price goes up or down.

  LSTM overcomes the vanishing gradient problem of simple RNNs via gated
  memory cells (Input gate, Forget gate, Output gate), allowing it to retain
  relevant patterns across the lookback window.

CLASSIFICATION CHANGES:
  * Output layer: sigmoid activation (0 = Down, 1 = Up probability).
  * Loss function: binary_crossentropy (directly optimises for correct
    directional prediction rather than minimising squared return error).
  * Smaller architecture: [32, 16] units. The original [128, 64] was
    over-parameterised for ~1500-row training sets and invited memorisation.
  * Shorter lookback: 30 days (market edge decays in ~2-4 weeks for daily data).

ARCHITECTURE:
  Input:          (batch, lookback=30, n_features)
  LSTM Layer 1:   32 units, return_sequences=True, dropout=0.4
  LSTM Layer 2:   16 units, return_sequences=False, dropout=0.4
  Dense:          16 units, ReLU, BatchNorm, Dropout=0.4
  Output:         1 unit, Sigmoid -> P(Up)
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    LSTM_LOOKBACK, LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS,
    LSTM_BATCH, LSTM_LR, LSTM_PATIENCE,
    TRAIN_END, VAL_END, OUTPUT_DIR, LOG_LEVEL, MODEL_MODE
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# DATA PREPARATION FOR LSTM
# -----------------------------------------------------------------------------

def prepare_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = None,
    lookback: int   = LSTM_LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Converts a flat DataFrame into 3-D sequences: (samples, timesteps, features).

    target_col defaults to 'target_direction' in classification mode.
    """
    if target_col is None:
        target_col = "target_direction" if MODEL_MODE == "classification" else "target_return"

    data    = df[feature_cols].values
    targets = df[target_col].values
    dates   = df.index

    X, y, idx = [], [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback: i])
        y.append(targets[i])
        idx.append(dates[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(idx)


def split_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = None,
    lookback: int = LSTM_LOOKBACK,
) -> Tuple:
    """Chrono train/val/test split for LSTM sequences."""
    X, y, dates = prepare_lstm_sequences(df, feature_cols, target_col=target_col, lookback=lookback)

    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask  = dates > VAL_END

    return (
        X[train_mask], y[train_mask], dates[train_mask],
        X[val_mask],   y[val_mask],   dates[val_mask],
        X[test_mask],  y[test_mask],  dates[test_mask],
    )


# -----------------------------------------------------------------------------
# MODEL DEFINITION
# -----------------------------------------------------------------------------

def build_lstm_model(
    n_features: int,
    lookback: int    = LSTM_LOOKBACK,
    units: List[int] = LSTM_UNITS,
    dropout: float   = LSTM_DROPOUT,
    lr: float        = LSTM_LR,
):
    """
    Build a stacked LSTM model for binary classification.

    ARCHITECTURE CHOICES:
      * Sigmoid output:  models P(Up direction) directly.
      * BinaryCrossentropy: optimal for balanced binary targets.
      * Smaller [32, 16] units: avoids memorising ~1500 training rows.
      * Higher dropout (0.4): strong regularisation signals.
      * Huber loss replaced with binary_crossentropy: classification
        loss is far better aligned with trading signal quality than MSE.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    if MODEL_MODE == "classification":
        output_activation = "sigmoid"
        loss_fn = keras.losses.BinaryCrossentropy()
        metrics_list = ["accuracy"]
    else:
        output_activation = "linear"
        loss_fn = keras.losses.Huber(delta=0.01)
        metrics_list = ["mae"]

    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),

        layers.LSTM(units[0], return_sequences=True,
                    dropout=dropout, recurrent_dropout=dropout / 2),
        layers.BatchNormalization(),

        layers.LSTM(units[1], return_sequences=False,
                    dropout=dropout, recurrent_dropout=dropout / 2),
        layers.BatchNormalization(),

        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(dropout),

        layers.Dense(1, activation=output_activation, name="output"),
    ], name="LSTM_DirectionPredictor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=metrics_list,
    )

    return model


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def train_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    ticker: str     = "STOCK",
    target_col: str = None,
    epochs: int     = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH,
    patience: int   = LSTM_PATIENCE,
) -> Tuple:
    """
    Full LSTM training pipeline with callbacks.

    CALLBACKS USED:
      * EarlyStopping(patience=15): stops when val_loss stops improving.
      * ReduceLROnPlateau: halves LR after 7 non-improving epochs.
      * ModelCheckpoint: saves the best-val-loss weights.

    Returns
    -------
    model        LSTM  -> captures sequential patterns; may overfit to trend direction
       RF    -> captures feature interactions; variance-reduction via bagging
       XGB   -> captures complex patterns; lower bias via boosting
       Each model makes different types of errors. Their combination
       averages out uncorrelated errors -> lower total generalisation error.
    history  : training history
    results  : dict with train/val/test metrics
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow not installed.")

    logger.info(f"[{ticker}] LSTM - Preparing sequences (lookback={LSTM_LOOKBACK}, mode={MODEL_MODE}, target={target_col or 'default'})...")

    splits = split_sequences(df, feature_cols, target_col=target_col)
    X_tr, y_tr, d_tr = splits[0], splits[1], splits[2]
    X_vl, y_vl, d_vl = splits[3], splits[4], splits[5]
    X_te, y_te, d_te = splits[6], splits[7], splits[8]

    logger.info(f"[{ticker}] Shapes - Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}")

    n_features = X_tr.shape[2]
    model = build_lstm_model(n_features)
    logger.info(model.summary())

    ckpt_path = MODEL_DIR / f"{ticker}_lstm_best.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict probabilities
    p_tr = model.predict(X_tr, verbose=0).flatten()
    p_vl = model.predict(X_vl, verbose=0).flatten()
    p_te = model.predict(X_te, verbose=0).flatten()

    results = {
        "train": _compute_metrics(y_tr, p_tr, "LSTM Train"),
        "val"  : _compute_metrics(y_vl, p_vl, "LSTM Val"),
        "test" : _compute_metrics(y_te, p_te, "LSTM Test"),
        "predictions": {
            "dates"    : d_te,
            "actual"   : y_te,
            "predicted": 2 * p_te - 1,   # [-1, 1] signal for backtester
            "proba"    : p_te,
        },
    }

    _log_metrics(ticker, "LSTM", results)
    _plot_training(history, ticker)
    _plot_predictions(results["predictions"], ticker, "LSTM")

    return model, history, results


# -----------------------------------------------------------------------------
# METRICS (internal helpers)
# -----------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> Dict:
    """
    In classification mode: Accuracy, ROC-AUC, F1
    In regression mode    : RMSE, MAE, R, DirAcc
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    if MODEL_MODE == "classification":
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = float(accuracy_score(y_true.astype(int), y_pred))
        try:
            auc = float(roc_auc_score(y_true.astype(int), y_prob))
        except Exception:
            auc = 0.5
        f1  = float(f1_score(y_true.astype(int), y_pred, zero_division=0))
        return {"label": label, "accuracy": acc, "roc_auc": auc, "f1": f1,
                "dir_acc": acc, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
    else:
        rmse    = float(np.sqrt(mean_squared_error(y_true, y_prob)))
        mae     = float(mean_absolute_error(y_true, y_prob))
        r2      = float(r2_score(y_true, y_prob))
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_prob)))
        return {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
                "dir_acc": dir_acc, "accuracy": dir_acc, "roc_auc": 0.0, "f1": 0.0}


def _log_metrics(ticker: str, model_name: str, results: Dict) -> None:
    for split in ["train", "val", "test"]:
        m = results[split]
        if MODEL_MODE == "classification":
            logger.info(
                f"[{ticker}] {model_name} {split.upper():5s} | "
                f"Acc={m['accuracy']:.3f} | AUC={m['roc_auc']:.3f} | F1={m['f1']:.3f}"
            )
        else:
            logger.info(
                f"[{ticker}] {model_name} {split.upper():5s} | "
                f"RMSE={m['rmse']:.5f} | MAE={m['mae']:.5f} | "
                f"R={m['r2']:.4f} | DirAcc={m['dir_acc']:.3f}"
            )


def _plot_training(history, ticker: str) -> None:
    metric = "accuracy" if MODEL_MODE == "classification" else "mae"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{ticker} - LSTM Loss ({'BinaryCE' if MODEL_MODE == 'classification' else 'Huber'})")
    axes[0].legend()
    if metric in history.history:
        axes[1].plot(history.history[metric],         label=f"Train {metric.title()}")
        axes[1].plot(history.history[f"val_{metric}"], label=f"Val {metric.title()}")
        axes[1].set_title(f"{ticker} - LSTM {metric.title()}")
        axes[1].legend()
        if MODEL_MODE == "classification":
            axes[1].axhline(0.5, color="red", linestyle="--", lw=0.8, label="Random")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_lstm_training.png", dpi=120)
    plt.close()


def _plot_predictions(preds: Dict, ticker: str, model_name: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(preds["dates"], preds["actual"],    label="Actual Direction",  lw=1.5)
    plt.plot(preds["dates"], preds["predicted"], label="Signal (2P-1)",     lw=1.0, alpha=0.8)
    plt.axhline(0, color="gray", linestyle="--", lw=0.8)
    plt.title(f"{ticker} - {model_name} | Test Set: Signal vs Actual Direction")
    plt.xlabel("Date")
    plt.ylabel("Direction / Signal [-1, 1]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_{model_name.lower()}_predictions.png", dpi=120)
    plt.close()
