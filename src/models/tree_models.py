"""
=============================================================================
STEP 4 & 5 - MODEL 2: RANDOM FOREST  |  MODEL 3: XGBOOST
=============================================================================

CLASSIFICATION MODE (default):
  Models now predict target_direction (1=Up, 0=Down) and output probabilities.
  Trading signal = 2 * P(Up) - 1, bounded in [-1, 1].

  WHY CLASSIFICATION > REGRESSION FOR TRADING:
    Predicting the exact magnitude of tomorrow's return (regression) is
    overwhelmingly noisy and models end up predicting values near zero, which
    yields terrible directional accuracy. Classification focuses the model
    on what actually matters for generating trades: "Will this go up or down?"

RANDOM FOREST - WHY:
  Random Forests handle tabular financial data exceptionally well because:
    1. Non-parametric: no distribution assumptions on returns.
    2. Feature interactions: tree splits capture interaction effects that
       linear models miss (e.g., "RSI is high AND volume is above average").
    3. Implicit feature selection: features with low predictive power are
       rarely chosen for splits -> built-in regularisation.
    4. Resistant to outliers: each tree sees a bootstrap subsample; extreme
       values influence fewer trees.
    5. Built-in OOB score: Out-Of-Bag error is an unbiased validation estimate.

XGBOOST - WHY:
  XGBoost is gradient boosted trees with several advantages over plain GBDT:
    1. Sequential correction: each tree corrects the residuals of the previous
       ensemble -> often achieves lower bias than RF.
    2. Regularisation (L1/L2): explicitly penalises complex trees.
    3. Handles missing values natively.
    4. Early stopping on validation set: stops before overfitting.

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_JOBS, RF_RANDOM_STATE,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE,
    XGB_COLSAMPLE, XGB_EARLY_STOPPING, XGB_RANDOM_STATE,
    TRAIN_END, VAL_END, OUTPUT_DIR, LOG_LEVEL, MODEL_MODE
)

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# SHARED SPLIT UTILITY
# -----------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = None,    # None = auto-detect by MODEL_MODE
) -> Tuple[np.ndarray, ...]:
    """
    Strict chronological train / val / test split.

    In classification mode uses target_direction; regression uses target_return.

    WHY NOT K-FOLD:
      Standard k-fold randomly shuffles rows. For time series, this causes
      future data to appear in training folds -> inflated performance metrics.
      Chronological split is the only valid approach.
    """
    if target_col is None:
        target_col = "target_direction" if MODEL_MODE == "classification" else "target_return"

    train_df = df[df.index <= TRAIN_END]
    val_df   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test_df  = df[df.index > VAL_END]

    X_tr = train_df[feature_cols].values
    y_tr = train_df[target_col].values
    X_vl = val_df[feature_cols].values
    y_vl = val_df[target_col].values
    X_te = test_df[feature_cols].values
    y_te = test_df[target_col].values

    return (X_tr, y_tr, train_df.index,
            X_vl, y_vl, val_df.index,
            X_te, y_te, test_df.index)


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred_or_prob: np.ndarray,
                    label: str = "", mode: str = MODEL_MODE) -> Dict:
    """
    Compute performance metrics depending on mode.

    Classification: Accuracy, ROC-AUC, F1, DirAcc
    Regression    : RMSE, MAE, R, DirAcc (kept for backward compat)
    """
    if mode == "classification":
        # y_pred_or_prob should be probability of class 1
        y_prob = y_pred_or_prob
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = float(accuracy_score(y_true, y_pred))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        f1  = float(f1_score(y_true, y_pred, zero_division=0))
        return {"label": label, "accuracy": acc, "roc_auc": auc, "f1": f1,
                "dir_acc": acc, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
    else:
        # regression
        y_pred = y_pred_or_prob
        rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae    = float(mean_absolute_error(y_true, y_pred))
        r2     = float(r2_score(y_true, y_pred))
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        return {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
                "dir_acc": dir_acc, "accuracy": dir_acc, "roc_auc": 0.0, "f1": 0.0}


def log_metrics(ticker: str, model_name: str, results: Dict) -> None:
    for split in ["train", "val", "test"]:
        m = results[split]
        if MODEL_MODE == "classification":
            logger.info(
                f"[{ticker}] {model_name:12s} {split.upper():5s} | "
                f"Acc={m['accuracy']:.3f} | AUC={m['roc_auc']:.3f} | F1={m['f1']:.3f}"
            )
        else:
            logger.info(
                f"[{ticker}] {model_name:12s} {split.upper():5s} | "
                f"RMSE={m['rmse']:.5f} | MAE={m['mae']:.5f} | "
                f"R={m['r2']:.4f} | DirAcc={m['dir_acc']:.3f}"
            )


# -----------------------------------------------------------------------------
# RANDOM FOREST
# -----------------------------------------------------------------------------

def train_random_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    ticker: str = "STOCK",
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest model (classifier by default).

    HYPERPARAMETERS JUSTIFICATION:
      n_estimators=500  : More trees -> lower variance. 500 is the point of
                          diminishing returns for most tabular problems.
      max_depth=None    : Full trees; depth controlled by min_samples_leaf.
      min_samples_leaf=5: Each leaf must have >=5 samples -> prevents memorising
                          individual days.
      n_jobs=-1         : Parallel training on all CPU cores.
    """
    logger.info(f"[{ticker}] Random Forest - Training ({MODEL_MODE} mode)...")
    splits = chronological_split(df, feature_cols)
    X_tr, y_tr, d_tr = splits[0], splits[1], splits[2]
    X_vl, y_vl, d_vl = splits[3], splits[4], splits[5]
    X_te, y_te, d_te = splits[6], splits[7], splits[8]

    logger.info(f"[{ticker}] RF Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}")

    if MODEL_MODE == "classification":
        rf = RandomForestClassifier(
            n_estimators     = RF_N_ESTIMATORS,
            max_depth        = RF_MAX_DEPTH,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            max_features     = "sqrt",
            n_jobs           = RF_N_JOBS,
            random_state     = RF_RANDOM_STATE,
            oob_score        = True,
            class_weight     = "balanced",   # handles class imbalance (more up-days)
        )
        rf.fit(X_tr, y_tr.astype(int))
        logger.info(f"[{ticker}] RF OOB Accuracy = {rf.oob_score_:.4f}")
        p_tr = rf.predict_proba(X_tr)[:, 1]
        p_vl = rf.predict_proba(X_vl)[:, 1]
        p_te = rf.predict_proba(X_te)[:, 1]
    else:
        rf = RandomForestRegressor(
            n_estimators     = RF_N_ESTIMATORS,
            max_depth        = RF_MAX_DEPTH,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            max_features     = "sqrt",
            n_jobs           = RF_N_JOBS,
            random_state     = RF_RANDOM_STATE,
            oob_score        = True,
        )
        rf.fit(X_tr, y_tr)
        logger.info(f"[{ticker}] RF OOB R2 = {rf.oob_score_:.4f}")
        p_tr = rf.predict(X_tr)
        p_vl = rf.predict(X_vl)
        p_te = rf.predict(X_te)

    results = {
        "train"   : compute_metrics(y_tr.astype(int), p_tr, "RF Train"),
        "val"     : compute_metrics(y_vl.astype(int), p_vl, "RF Val"),
        "test"    : compute_metrics(y_te.astype(int), p_te, "RF Test"),
        "oob_score": rf.oob_score_,
        # signal = 2*P(Up)-1  [-1, 1] for the backtester
        "predictions": {
            "dates": d_te,
            "actual": y_te,
            "predicted": p_te if MODEL_MODE == "regression" else 2 * p_te - 1,
            "proba": p_te,
        },
    }
    log_metrics(ticker, "RandomForest", results)

    _plot_rf_importance(rf, feature_cols, ticker)
    _plot_predictions(results["predictions"], ticker, "RF")

    return rf, results


def _plot_rf_importance(
    model: RandomForestClassifier,
    feature_cols: List[str],
    ticker: str,
    top_n: int = 30,
) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{ticker} - Random Forest Feature Importances (Top {top_n})")
    ax.set_xlabel("Importance (Mean Decrease in Impurity)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_rf_importance.png", dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# XGBOOST
# -----------------------------------------------------------------------------

def train_xgboost(
    df: pd.DataFrame,
    feature_cols: List[str],
    ticker: str = "STOCK",
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train an XGBoost gradient-boosted trees model (classifier by default).

    KEY CHANGES vs REGRESSION:
      eval_metric = "logloss"  - optimises directly for classification quality.
      max_depth   = 3          - shallower trees, better generalisation on noisy data.
      colsample   = 0.5        - forces use of random feature subset, decorrelates trees.
      scale_pos_weight set to handle class imbalance.
    """
    logger.info(f"[{ticker}] XGBoost - Training ({MODEL_MODE} mode)...")
    splits = chronological_split(df, feature_cols)
    X_tr, y_tr, d_tr = splits[0], splits[1], splits[2]
    X_vl, y_vl, d_vl = splits[3], splits[4], splits[5]
    X_te, y_te, d_te = splits[6], splits[7], splits[8]

    logger.info(f"[{ticker}] XGB Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}")

    if MODEL_MODE == "classification":
        # Handle class imbalance
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        scale_pos = n_neg / max(n_pos, 1)

        xgb_model = xgb.XGBClassifier(
            n_estimators          = XGB_N_ESTIMATORS,
            max_depth             = XGB_MAX_DEPTH,
            learning_rate         = XGB_LEARNING_RATE,
            subsample             = XGB_SUBSAMPLE,
            colsample_bytree      = XGB_COLSAMPLE,
            reg_alpha             = 0.1,          # L1 regularisation (increased)
            reg_lambda            = 1.5,          # L2 regularisation (increased)
            tree_method           = "hist",
            eval_metric           = "logloss",
            early_stopping_rounds = XGB_EARLY_STOPPING,
            random_state          = XGB_RANDOM_STATE,
            scale_pos_weight      = scale_pos,    # balance classes
            use_label_encoder     = False,
            verbosity             = 1,
        )
        xgb_model.fit(
            X_tr, y_tr.astype(int),
            eval_set=[(X_vl, y_vl.astype(int))],
            verbose=50,
        )
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators          = XGB_N_ESTIMATORS,
            max_depth             = XGB_MAX_DEPTH,
            learning_rate         = XGB_LEARNING_RATE,
            subsample             = XGB_SUBSAMPLE,
            colsample_bytree      = XGB_COLSAMPLE,
            reg_alpha             = 0.1,
            reg_lambda            = 1.5,
            tree_method           = "hist",
            eval_metric           = "rmse",
            early_stopping_rounds = XGB_EARLY_STOPPING,
            random_state          = XGB_RANDOM_STATE,
            verbosity             = 1,
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=50,
        )

    logger.info(f"[{ticker}] XGBoost best iteration: {xgb_model.best_iteration}")

    if MODEL_MODE == "classification":
        p_tr = xgb_model.predict_proba(X_tr)[:, 1]
        p_vl = xgb_model.predict_proba(X_vl)[:, 1]
        p_te = xgb_model.predict_proba(X_te)[:, 1]
    else:
        p_tr = xgb_model.predict(X_tr)
        p_vl = xgb_model.predict(X_vl)
        p_te = xgb_model.predict(X_te)

    results = {
        "train"         : compute_metrics(y_tr.astype(int), p_tr, "XGB Train"),
        "val"           : compute_metrics(y_vl.astype(int), p_vl, "XGB Val"),
        "test"          : compute_metrics(y_te.astype(int), p_te, "XGB Test"),
        "best_iteration": xgb_model.best_iteration,
        "predictions"   : {
            "dates": d_te,
            "actual": y_te,
            "predicted": p_te if MODEL_MODE == "regression" else 2 * p_te - 1,   # [-1, 1] signal
            "proba": p_te,
        },
    }
    log_metrics(ticker, "XGBoost", results)

    _plot_xgb_importance(xgb_model, feature_cols, ticker)
    _plot_predictions(results["predictions"], ticker, "XGBoost")

    return xgb_model, results


def _plot_xgb_importance(
    model: xgb.XGBClassifier,
    feature_cols: List[str],
    ticker: str,
    top_n: int = 30,
) -> None:
    scores = model.get_booster().get_score(importance_type="gain")
    importance = pd.Series(scores).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    importance.sort_values().plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(f"{ticker} - XGBoost Feature Importances (Gain, Top {top_n})")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_xgb_importance.png", dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# SHARED PLOT
# -----------------------------------------------------------------------------

def _plot_predictions(preds: Dict, ticker: str, model_name: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(preds["dates"], preds["actual"],    label="Actual Direction",  lw=1.5)
    plt.plot(preds["dates"], preds["predicted"], label="Signal (2P-1)",     lw=1.0, alpha=0.8)
    plt.axhline(0, color="gray", linestyle="--", lw=0.8)
    plt.title(f"{ticker} - {model_name} | Test Set: Signal vs Actual Direction")
    plt.xlabel("Date")
    plt.ylabel("Direction / Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_{model_name.lower()}_predictions.png", dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# COMPARISON
# -----------------------------------------------------------------------------

def compare_models(
    rf_results: Dict,
    xgb_results: Dict,
    lstm_results: Optional[Dict] = None,
    hybrid_results: Optional[Dict] = None,
    ticker: str = "STOCK",
) -> pd.DataFrame:
    """
    Produce a side-by-side comparison table of all models on the TEST set.

    In classification mode, the primary metric is ROC-AUC (preferred over raw
    accuracy as it accounts for class imbalance and probability calibration).
    """
    rows = []
    for name, res in [("RandomForest", rf_results),
                       ("XGBoost",      xgb_results),
                       ("LSTM",         lstm_results),
                       ("Hybrid",       hybrid_results)]:
        if res is None:
            continue
        m = res["test"]
        if MODEL_MODE == "classification":
            rows.append({
                "Model"  : name,
                "Accuracy": round(m.get("accuracy", 0), 4),
                "ROC_AUC": round(m.get("roc_auc", 0), 4),
                "F1"     : round(m.get("f1", 0), 4),
                "DirAcc" : round(m.get("dir_acc", 0), 4),
                # Keep RMSE/R columns for compatibility with API
                "RMSE"   : round(m.get("rmse", 0), 5),
                "MAE"    : round(m.get("mae", 0), 5),
                "R"     : round(m.get("r2", 0), 4),
            })
        else:
            rows.append({
                "Model"  : name,
                "RMSE"   : round(m["rmse"],    5),
                "MAE"    : round(m["mae"],     5),
                "R"     : round(m["r2"],      4),
                "DirAcc" : round(m["dir_acc"], 4),
            })

    comparison = pd.DataFrame(rows).set_index("Model")
    logger.info(f"\n[{ticker}] === MODEL COMPARISON (Test Set) ===\n{comparison}\n")

    # Plots
    if MODEL_MODE == "classification":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        comparison[["Accuracy", "ROC_AUC", "F1"]].plot(kind="bar", ax=axes[0], colormap="Set2")
        axes[0].set_title(f"{ticker} - Classification Metrics (higher is better)")
        axes[0].set_xticklabels(comparison.index, rotation=30)
        axes[0].axhline(0.5, color="red", linestyle="--", lw=0.8, label="Random baseline")
        comparison[["DirAcc"]].plot(kind="bar", ax=axes[1], colormap="Set1")
        axes[1].set_title(f"{ticker} - Directional Accuracy")
        axes[1].set_xticklabels(comparison.index, rotation=30)
        axes[1].axhline(0.5, color="red", linestyle="--", lw=0.8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        comparison[["RMSE", "MAE"]].plot(kind="bar", ax=axes[0], colormap="Set2")
        axes[0].set_title(f"{ticker} - RMSE & MAE (lower is better)")
        axes[0].set_xticklabels(comparison.index, rotation=30)
        comparison[["R", "DirAcc"]].plot(kind="bar", ax=axes[1], colormap="Set1")
        axes[1].set_title(f"{ticker} - R & Directional Accuracy (higher is better)")
        axes[1].set_xticklabels(comparison.index, rotation=30)

    plt.suptitle(f"{ticker} - Individual Model Performance Comparison ({MODEL_MODE.title()})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ticker}_model_comparison.png", dpi=120)
    plt.close()

    return comparison
