"""
=============================================================================
STEP 6 - ENSEMBLE MODEL
=============================================================================
WHY ENSEMBLE AFTER INDIVIDUAL EVALUATION:
  We first evaluated each model independently to understand:
    * Which model is strongest overall (best DirAcc on test set)
    * Which model is strongest on specific market regimes (trending vs choppy)
    * Where models agree and disagree (high disagreement = high uncertainty)

  Now we combine them because:
    1. BIAS-VARIANCE DECOMPOSITION:
       LSTM  -> captures sequential patterns; may overfit to trend direction
       RF    -> captures feature interactions; variance-reduction via bagging
       XGB   -> captures complex patterns; lower bias via boosting
       Each model makes different types of errors. Their combination
       averages out uncorrelated errors -> lower total generalisation error.

    2. EMPIRICAL EVIDENCE:
       Ensemble methods win most ML competitions (Kaggle) and have been shown
       in academic literature (Geurts, Ernst 2006; Chen & Guestrin 2016) to
       outperform any single model when base models are diverse.

    3. DIVERSIFICATION PRINCIPLE (from portfolio theory):
       Just as diversifying assets reduces portfolio risk, diversifying model
       predictions reduces prediction risk.

THREE ENSEMBLE STRATEGIES IMPLEMENTED:
  1. Weighted Averaging : Simple, interpretable. Weights set by validation perf.
  2. Blending          : Hold-out validation set predictions as meta-features.
  3. Stacking          : Meta-learner (Ridge) trained on out-of-fold predictions
                         from base models. Most powerful but needs sufficient data.
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score,
)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    ENSEMBLE_METHOD, ENSEMBLE_WEIGHTS,
    TRAIN_END, VAL_END, OUTPUT_DIR, LOG_LEVEL, MODEL_MODE,
    RF_MIN_SAMPLES_LEAF, RF_RANDOM_STATE
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------------

def _metrics(y_true, y_pred_or_prob, label="") -> Dict:
    """
    Classification mode: interpret y_pred_or_prob as probability of class 1.
    Regression mode    : interpret y_pred_or_prob as continuous prediction.
    """
    if MODEL_MODE == "classification":
        y_prob = y_pred_or_prob
        y_pred = (y_prob >= 0.5).astype(int)
        acc  = float(accuracy_score(y_true.astype(int), y_pred))
        try:
            auc = float(roc_auc_score(y_true.astype(int), y_prob))
        except Exception:
            auc = 0.5
        f1  = float(f1_score(y_true.astype(int), y_pred, zero_division=0))
        return {"label": label, "accuracy": acc, "roc_auc": auc, "f1": f1,
                "dir_acc": acc, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
    else:
        rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred_or_prob)))
        mae     = float(mean_absolute_error(y_true, y_pred_or_prob))
        r2      = float(r2_score(y_true, y_pred_or_prob))
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred_or_prob)))
        return {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
                "dir_acc": dir_acc, "accuracy": dir_acc, "roc_auc": 0.0, "f1": 0.0}


# -----------------------------------------------------------------------------
# METHOD 1: WEIGHTED AVERAGING
# -----------------------------------------------------------------------------

def weighted_average_ensemble(
    predictions: Dict[str, np.ndarray],   # model_name -> probabilities P(Up)
    y_true: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    ticker: str = "STOCK",
) -> Tuple[np.ndarray, Dict]:
    """
    In classification mode, blend predicted probabilities P(Up) from each model.
    Final signal = 2 * blended_prob - 1   [-1, 1].
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    # Normalise weights
    total = sum(weights.get(k, 0) for k in predictions)
    norm_weights = {k: weights.get(k, 0) / total for k in predictions}

    logger.info(f"[{ticker}] Ensemble weights: {norm_weights}")

    # Blend probabilities
    blended_prob = sum(norm_weights[k] * v for k, v in predictions.items())

    if MODEL_MODE == "classification":
        # Output a blended prob for metrics, but also create signal
        metrics = _metrics(y_true, blended_prob, "Weighted Ensemble Test")
        y_signal = 2 * blended_prob - 1   # [-1, 1] signal for backtester
        logger.info(
            f"[{ticker}] Weighted Ensemble | Acc={metrics['accuracy']:.3f} | "
            f"AUC={metrics['roc_auc']:.3f} | F1={metrics['f1']:.3f}"
        )
    else:
        y_signal = blended_prob   # already continuous signal
        metrics  = _metrics(y_true, y_signal, "Weighted Ensemble Test")
        logger.info(
            f"[{ticker}] Weighted Ensemble | RMSE={metrics['rmse']:.5f} | "
            f"MAE={metrics['mae']:.5f} | R={metrics['r2']:.4f} | DirAcc={metrics['dir_acc']:.3f}"
        )

    return y_signal, metrics


# -----------------------------------------------------------------------------
# METHOD 2: BLENDING
# -----------------------------------------------------------------------------

def blending_ensemble(
    val_predictions: Dict[str, np.ndarray],
    test_predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    y_test: np.ndarray,
    ticker: str = "STOCK",
) -> Tuple[np.ndarray, Dict, object]:
    """
    In classification mode: meta-learner is LogisticRegression trained on
    base model probability columns. Outputs probability, then mapped to signal.
    In regression mode: Ridge meta-learner as before.
    """
    from sklearn.linear_model import LogisticRegression

    X_meta_val  = np.column_stack([val_predictions[k]  for k in sorted(val_predictions)])
    X_meta_test = np.column_stack([test_predictions[k] for k in sorted(test_predictions)])

    model_names = sorted(val_predictions.keys())
    logger.info(f"[{ticker}] Blending meta-learner input features: {model_names}")

    if MODEL_MODE == "classification":
        meta_learner = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        meta_learner.fit(X_meta_val, y_val.astype(int))
        y_blend_test_prob = meta_learner.predict_proba(X_meta_test)[:, 1]
        y_blend_val_prob  = meta_learner.predict_proba(X_meta_val)[:, 1]
        val_metrics  = _metrics(y_val,  y_blend_val_prob,  "Blend Val")
        test_metrics = _metrics(y_test, y_blend_test_prob, "Blend Test")
        y_signal = 2 * y_blend_test_prob - 1   # map to [-1, 1]
        logger.info(
            f"[{ticker}] Blending Ensemble | Acc={test_metrics['accuracy']:.3f} | "
            f"AUC={test_metrics['roc_auc']:.3f}"
        )
    else:
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(X_meta_val, y_val)
        y_blend_val  = meta_learner.predict(X_meta_val)
        y_blend_test = meta_learner.predict(X_meta_test)
        val_metrics  = _metrics(y_val,  y_blend_val,  "Blend Val")
        test_metrics = _metrics(y_test, y_blend_test, "Blend Test")
        y_signal = y_blend_test
        logger.info(
            f"[{ticker}] Blending Ensemble | RMSE={test_metrics['rmse']:.5f} | "
            f"R={test_metrics['r2']:.4f} | DirAcc={test_metrics['dir_acc']:.3f}"
        )

    return y_signal, test_metrics, meta_learner


# -----------------------------------------------------------------------------
# METHOD 3: STACKING (OOF)
# -----------------------------------------------------------------------------

def stacking_ensemble(
    df: pd.DataFrame,
    feature_cols: List[str],
    rf_model,
    xgb_model,
    lstm_model,      # can be None
    lstm_lookback: int,
    ticker: str = "STOCK",
) -> Tuple[np.ndarray, Dict]:
    """
    OUT-OF-FOLD (OOF) STACKING:
        1. Split training data into K folds (K=5, time-series aware).
        2. For each fold: train base models on K-1 folds, predict on fold K.
        3. Collect OOF predictions for ALL training samples.
        4. Train meta-learner on OOF predictions (avoids data leakage).
        5. Base models retrained on FULL training set.
        6. Meta-learner predicts on test using full-model predictions.

    WHY OOF > BLENDING:
        Blending uses only the validation set to train the meta-learner
        (~1 year of data here). OOF uses the ENTIRE training set (6 years)
        for meta-learner training -> much more reliable coefficient estimates.

    LIMITATION:
        Computationally expensive: K x (RF + XGB + optional LSTM) training runs.
        LSTM stacking is approximated by using its test-set predictions directly
        since full OOF LSTM training would be prohibitively slow.

    NOTE: For this implementation we show RF + XGB stacking with Ridge meta-learner.
    LSTM predictions are included as a third column if available.
    """
    logger.info(f"[{ticker}] Stacking Ensemble - Generating OOF predictions...")

    train_df = df[df.index <= TRAIN_END].dropna(subset=["target_return"])
    test_df  = df[df.index > VAL_END].dropna(subset=["target_return"])

    X_train = train_df[feature_cols].values
    y_train = train_df["target_return"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target_return"].values

    n_folds = 5
    # Time-series K-fold: always respect chronological order
    fold_size = len(X_train) // n_folds

    oof_rf  = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < n_folds - 1 else len(X_train)
        tr_idx    = list(range(0, val_start))
        vl_idx    = list(range(val_start, val_end))

        if len(tr_idx) == 0:
            continue

        def get_prob_up(model, X_val):
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 1:
                # Only one class was present in the training fold
                single_class = model.classes_[0]
                return np.ones(len(X_val)) if single_class == 1 else np.zeros(len(X_val))
            return probs[:, 1]

        rf_fold = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features="sqrt", n_jobs=-1, random_state=RF_RANDOM_STATE,
            class_weight="balanced")
        rf_fold.fit(X_train[tr_idx], y_train[tr_idx].astype(int))
        oof_rf[vl_idx] = get_prob_up(rf_fold, X_train[vl_idx]) if MODEL_MODE == "classification" else rf_fold.predict(X_train[vl_idx])

        # XGB fold
        import xgboost as xgb_lib
        from configs.config import XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE
        xgb_fold = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, tree_method="hist",
            eval_metric="logloss", random_state=XGB_RANDOM_STATE, verbosity=0)
        xgb_fold.fit(X_train[tr_idx], y_train[tr_idx].astype(int))
        oof_xgb[vl_idx] = get_prob_up(xgb_fold, X_train[vl_idx]) if MODEL_MODE == "classification" else xgb_fold.predict(X_train[vl_idx])

        logger.info(f"[{ticker}] OOF fold {fold+1}/{n_folds} done.")

    # Test set predictions from full models (probabilities in classification mode)
    test_rf  = rf_model.predict_proba(X_test)[:, 1] if MODEL_MODE == "classification" else rf_model.predict(X_test)
    test_xgb = xgb_model.predict_proba(X_test)[:, 1] if MODEL_MODE == "classification" else xgb_model.predict(X_test)

    # Build meta-feature matrices
    X_meta_train = np.column_stack([oof_rf, oof_xgb])
    X_meta_test  = np.column_stack([test_rf, test_xgb])

    # Include LSTM predictions if available
    if lstm_model is not None:
        try:
            from src.models.lstm_model import prepare_lstm_sequences
            X_seq, y_seq, dates_seq = prepare_lstm_sequences(
                df[df.index > VAL_END].dropna(subset=["target_return"]),
                feature_cols, lookback=lstm_lookback)
            test_lstm = lstm_model.predict(X_seq, verbose=0).flatten()
            # Align lengths (LSTM loses `lookback` rows)
            min_len = min(len(test_xgb), len(test_lstm))
            X_meta_test = np.column_stack([
                test_rf[-min_len:], test_xgb[-min_len:], test_lstm])
            X_meta_train = X_meta_train  # keep as-is (LSTM OOF not computed)
            y_test = y_test[-min_len:]
            logger.info(f"[{ticker}] LSTM included in stacking meta-features.")
        except Exception as e:
            logger.warning(f"[{ticker}] LSTM stacking skipped: {e}")

    # Train meta-learner on OOF
    if MODEL_MODE == "classification":
        from sklearn.linear_model import LogisticRegression
        y_tr_int = y_train.astype(int)
        
        # Check if we only have one class to train on (edge case in imbalanced sequential splits)
        if len(np.unique(y_tr_int)) == 1:
            logger.warning(f"[{ticker}] Stacking meta-learner found only 1 class in y_train. Bypassing LR.")
            single_c = y_tr_int[0]
            y_stack_prob = np.ones(len(X_meta_test)) if single_c == 1 else np.zeros(len(X_meta_test))
        else:
            meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            meta.fit(X_meta_train, y_tr_int)
            y_stack_prob = meta.predict_proba(X_meta_test)[:, 1]
            
        y_stack      = 2 * y_stack_prob - 1    # [-1, 1] trading signal
        test_metrics = _metrics(y_test, y_stack_prob, "Stacking Test")
        logger.info(
            f"[{ticker}] Stacking Ensemble | Acc={test_metrics['accuracy']:.4f} | "
            f"AUC={test_metrics['roc_auc']:.4f}"
        )
        if len(np.unique(y_tr_int)) > 1:
            logger.info(f"[{ticker}] Stacking meta-learner coefs: {meta.coef_.round(4)}")
    else:
        meta = Ridge(alpha=1.0)
        meta.fit(X_meta_train, y_train)
        y_stack = meta.predict(X_meta_test)
        test_metrics = _metrics(y_test, y_stack, "Stacking Test")
        logger.info(
            f"[{ticker}] Stacking Ensemble | RMSE={test_metrics['rmse']:.5f} | "
            f"R={test_metrics['r2']:.4f} | DirAcc={test_metrics['dir_acc']:.3f}"
        )
        logger.info(f"[{ticker}] Stacking meta-learner coefs: {meta.coef_.round(4)}")

    return y_stack, test_metrics


# -----------------------------------------------------------------------------
# ENSEMBLE COORDINATOR
# -----------------------------------------------------------------------------

def run_ensemble(
    df: pd.DataFrame,
    feature_cols: List[str],
    rf_model,
    xgb_model,
    rf_results: Dict,
    xgb_results: Dict,
    lstm_model    = None,
    lstm_results: Optional[Dict] = None,
    method: str   = ENSEMBLE_METHOD,
    ticker: str   = "STOCK",
    lstm_lookback: int = 60,
) -> Tuple[np.ndarray, Dict]:
    """
    Dispatch to the chosen ensemble method and return final predictions + metrics.
    """
    logger.info(f"\n{'='*60}\nSTEP 6 - ENSEMBLE ({method.upper()}) for {ticker}\n{'='*60}")

    test_df = df[df.index > VAL_END].dropna(subset=["target_return"])
    val_df  = df[(df.index > TRAIN_END) & (df.index <= VAL_END)].dropna(subset=["target_return"])

    feat_arr_test = test_df[feature_cols].values
    feat_arr_val  = val_df[feature_cols].values
    y_test        = test_df["target_return"].values
    y_val         = val_df["target_return"].values

    # Use probabilities for ensemble blending (classification) or direct preds (regression)
    if MODEL_MODE == "classification":
        test_preds = {
            "rf"  : rf_model.predict_proba(feat_arr_test)[:, 1],
            "xgb" : xgb_model.predict_proba(feat_arr_test)[:, 1],
        }
        val_preds = {
            "rf"  : rf_model.predict_proba(feat_arr_val)[:, 1],
            "xgb" : xgb_model.predict_proba(feat_arr_val)[:, 1],
        }
    else:
        test_preds = {
            "rf"  : rf_model.predict(feat_arr_test),
            "xgb" : xgb_model.predict(feat_arr_test),
        }
        val_preds = {
            "rf"  : rf_model.predict(feat_arr_val),
            "xgb" : xgb_model.predict(feat_arr_val),
        }

    if lstm_model is not None and lstm_results is not None:
        test_preds["lstm"] = lstm_results["predictions"]["predicted"]
        val_preds["lstm"]  = None  # blending needs val lstm preds - skip for now

    if method == "weighted":
        y_final, metrics = weighted_average_ensemble(test_preds, y_test, ticker=ticker)

    elif method == "blending":
        # Exclude LSTM from blending (requires aligned val predictions)
        val_preds_clean  = {k: v for k, v in val_preds.items() if v is not None}
        test_preds_clean = {k: test_preds[k] for k in val_preds_clean}
        y_final, metrics, _ = blending_ensemble(
            val_preds_clean, test_preds_clean, y_val, y_test, ticker=ticker)

    elif method == "stacking":
        y_final, metrics = stacking_ensemble(
            df, feature_cols, rf_model, xgb_model, lstm_model, lstm_lookback, ticker=ticker)
        y_test = test_df["target_return"].values[-len(y_final):]  # align

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # Compare ensemble vs best individual
    best_individual_dir = max(
        rf_results["test"]["dir_acc"],
        xgb_results["test"]["dir_acc"],
        (lstm_results["test"]["dir_acc"] if lstm_results else 0),
    )
    if MODEL_MODE == "classification":
        best_metric = max(
            rf_results["test"].get("roc_auc", 0),
            xgb_results["test"].get("roc_auc", 0),
            (lstm_results["test"].get("roc_auc", 0) if lstm_results else 0),
        )
        logger.info(
            f"[{ticker}] Ensemble Acc={metrics['accuracy']:.4f} / AUC={metrics.get('roc_auc', 0):.4f} "
            f"vs Best Individual AUC={best_metric:.4f}"
        )
    else:
        logger.info(
            f"[{ticker}] Ensemble DirAcc={metrics['dir_acc']:.4f} "
            f"vs Best Individual DirAcc={best_individual_dir:.4f} "
            f"| Improvement={metrics['dir_acc'] - best_individual_dir:+.4f}"
        )

    _plot_ensemble(test_df.index[-len(y_final):], y_test, y_final, ticker, method)

    return y_final, metrics


def _plot_ensemble(dates, y_true, y_pred, ticker, method) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true, label="Actual", lw=1.5)
    plt.plot(dates, y_pred, label=f"Ensemble ({method})", lw=1.2, color="crimson")
    plt.title(f"{ticker} - Ensemble Prediction vs Actual (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_ensemble_predictions.png", dpi=120)
    plt.close()
