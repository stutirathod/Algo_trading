"""
=============================================================================
FLASK API - connects pipeline outputs -> React dashboard
=============================================================================
Run:  python app.py
Open: http://localhost:5000

Reads from outputs/ folder (produced by pipeline.py --mode train)
Also exposes /api/run-pipeline to trigger training from the UI
=============================================================================
"""

import glob
import json
import os
import pickle
import subprocess
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR  = BASE_DIR / "data" / "cache"
PROC_DIR   = BASE_DIR / "data" / "processed"
ART_DIR    = OUTPUT_DIR / "artifacts"
BT_DIR     = OUTPUT_DIR / "backtest"

app = Flask(__name__, static_folder="dashboard/build", static_url_path="")
CORS(app)   # allow React dev-server on :3000 to call :5000

# -- in-memory pipeline state --------------------------------------------------
pipeline_state = {
    "running": False,
    "log": [],
    "progress": 0,
    "started_at": None,
}

# =============================================================================
# HELPERS
# =============================================================================

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _jsonify_df(df: pd.DataFrame, orient="records") -> list:
    """Convert a DataFrame to JSON-safe list, handling NaN/Inf."""
    if df.empty:
        return []
    return json.loads(
        df.replace([np.inf, -np.inf], np.nan)
          .fillna("null")
          .to_json(orient=orient, date_format="iso")
    )


def _ticker_slug(ticker: str) -> str:
    return ticker.replace(".", "_")


def _available_tickers() -> list:
    """Detect which tickers have been trained by checking artifacts."""
    ART_DIR.mkdir(parents=True, exist_ok=True)
    rf_files = list(ART_DIR.glob("*_rf.pkl"))
    tickers  = [f.stem.replace("_rf", "").replace("_", ".") for f in rf_files]
    # Fix double-dot edge case e.g. RELIANCE_NS -> RELIANCE.NS
    tickers  = [t.replace("_NS", ".NS").replace("_BO", ".BO") for t in tickers]
    return tickers if tickers else ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


# =============================================================================
# API ROUTES
# =============================================================================

@app.route("/api/tickers")
def get_tickers():
    """List tickers that have completed training."""
    return jsonify(_available_tickers())


# --- Price / OHLCV ------------------------------------------------------------

@app.route("/api/price/<ticker>")
def get_price(ticker):
    """
    Return OHLCV + key technical columns for a ticker.
    Source: data/processed/<ticker>_features.parquet (built by engineer.py)
    Falls back to raw cache parquet if processed not available.
    """
    slug = _ticker_slug(ticker)

    # Try processed features first (has all technical indicators)
    proc_path = PROC_DIR / f"{slug}_features.parquet"
    raw_path  = CACHE_DIR / f"{slug}_price.parquet"

    df = _safe_read_parquet(proc_path)
    if df.empty:
        df = _safe_read_parquet(raw_path)

    if df.empty:
        return jsonify({"error": f"No price data for {ticker}"}), 404

    # Select columns available
    keep = ["Open", "High", "Low", "Close", "Volume",
            "sma_20", "sma_50", "sma_200",
            "bb_upper", "bb_lower", "bb_mid",
            "rsi", "macd", "macd_signal", "macd_diff",
            "realised_vol", "atr_pct",
            "vwap_20", "obv"]
    available = [c for c in keep if c in df.columns]
    out = df[available].tail(520).reset_index()
    out.rename(columns={"index": "date", "Date": "date"}, inplace=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    return jsonify(_jsonify_df(out))


# --- Model Metrics ------------------------------------------------------------

@app.route("/api/metrics/<ticker>")
def get_metrics(ticker):
    """
    Return per-model RMSE / MAE / R / DirAcc for all splits.
    Source: outputs/<ticker>_model_comparison.csv
    """
    slug = _ticker_slug(ticker)
    path = OUTPUT_DIR / f"{slug}_model_comparison.csv"
    df   = _safe_read_csv(path)

    if df.empty:
        return jsonify({"error": "No model metrics found. Run pipeline first."}), 404

    # Reshape: rows=models, cols=metrics
    result = {}
    for _, row in df.iterrows():
        model = str(row.get("Model", row.name)).lower()
        result[model] = {
            "rmse"    : float(row.get("RMSE",     0)),
            "mae"     : float(row.get("MAE",      0)),
            "r2"      : float(row.get("R",       0)),
            "dir_acc" : float(row.get("DirAcc",   0)),
            "accuracy": float(row.get("Accuracy", row.get("DirAcc", 0))),
            "roc_auc" : float(row.get("ROC_AUC",  0)),
            "f1"      : float(row.get("F1",       0)),
        }
    return jsonify(result)


# --- Backtest results ---------------------------------------------------------

@app.route("/api/backtest/<ticker>")
def get_backtest(ticker):
    """
    Return backtest performance metrics + equity curve.
    Sources:
      outputs/backtest/<ticker>_*_metrics.csv   -> scalar metrics
      outputs/backtest/<ticker>_equity.csv       -> time series (if saved)
    """
    slug  = _ticker_slug(ticker)

    # Scalar metrics - try any strategy name
    metric_files = list(BT_DIR.glob(f"{slug}_*_metrics.csv"))
    metrics = {}
    if metric_files:
        mdf = pd.read_csv(metric_files[0])
        if not mdf.empty:
            metrics = mdf.iloc[0].to_dict()

    # Equity curve CSV (written by save_equity_curve below)
    eq_path = BT_DIR / f"{slug}_equity.csv"
    equity  = []
    if eq_path.exists():
        edf = pd.read_csv(eq_path)
        edf["date"] = pd.to_datetime(edf.get("date", edf.index)).dt.strftime("%Y-%m-%d")
        equity = _jsonify_df(edf)

    return jsonify({"metrics": metrics, "equity_curve": equity})


# --- Signals -----------------------------------------------------------------

@app.route("/api/signals")
def get_signals():
    """
    Return today's signals across all tickers.
    Source: outputs/signals_YYYY-MM-DD.csv  (written by pipeline predict mode)
    Falls back to most recent signals file if today's doesn't exist.
    """
    today     = date.today().strftime("%Y-%m-%d")
    sig_path  = OUTPUT_DIR / f"signals_{today}.csv"

    if not sig_path.exists():
        # Find most recent
        all_sig = sorted(OUTPUT_DIR.glob("signals_*.csv"), reverse=True)
        if all_sig:
            sig_path = all_sig[0]
        else:
            return jsonify([])

    df = pd.read_csv(sig_path)
    return jsonify(_jsonify_df(df))


# --- Feature importance -------------------------------------------------------

@app.route("/api/features/<ticker>")
def get_features(ticker):
    """
    Return feature importance from RF and XGBoost models.
    Sources: saved model artifacts in outputs/artifacts/
    """
    slug     = _ticker_slug(ticker)
    rf_path  = ART_DIR / f"{slug}_rf.pkl"
    xgb_path = ART_DIR / f"{slug}_xgb.json"

    # Load scaler to get feature names
    scaler_path = ART_DIR / f"{slug}_scaler.pkl"
    feature_names = []
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        feature_names = scaler.get("feature_cols", [])

    rf_importance  = {}
    xgb_importance = {}

    if rf_path.exists() and feature_names:
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        rf_importance = dict(zip(
            feature_names,
            rf.feature_importances_.tolist()
        ))

    if xgb_path.exists():
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(xgb_path))
        scores = model.get_booster().get_score(importance_type="gain")
        xgb_importance = {k: float(v) for k, v in scores.items()}

    # Merge and return top 20
    all_features = set(list(rf_importance.keys()) + list(xgb_importance.keys()))
    rows = []
    for feat in all_features:
        rf_val  = rf_importance.get(feat, 0)
        xgb_val = xgb_importance.get(feat, 0)
        if rf_val > 0 or xgb_val > 0:
            rows.append({"feature": feat, "rf": round(rf_val, 6), "xgb": round(xgb_val, 6)})

    rows.sort(key=lambda r: r["rf"] + r["xgb"], reverse=True)
    return jsonify(rows[:20])


# --- EDA summary -------------------------------------------------------------

@app.route("/api/eda/<ticker>")
def get_eda(ticker):
    """
    Return EDA statistics for a ticker.
    Computed live from processed parquet (fast - no model needed).
    """
    slug  = _ticker_slug(ticker)
    path  = PROC_DIR / f"{slug}_features.parquet"
    df    = _safe_read_parquet(path)
    if df.empty:
        path = CACHE_DIR / f"{slug}_price.parquet"
        df   = _safe_read_parquet(path)
    if df.empty:
        return jsonify({"error": "No data"}), 404

    close   = df["Close"].dropna()
    returns = close.pct_change().dropna()

    from scipy import stats as sp_stats
    jb_stat, jb_p = sp_stats.jarque_bera(returns)
    ann_vol = float(returns.std() * np.sqrt(252))

    # Return distribution histogram buckets
    counts, edges = np.histogram(returns, bins=60)
    hist = [{"x": round(float((edges[i] + edges[i+1]) / 2), 5), "count": int(counts[i])}
            for i in range(len(counts))]

    # Rolling 20-day vol
    roll_vol = returns.rolling(20).std() * np.sqrt(252)
    vol_ts = []
    for d, v in roll_vol.dropna().items():
        vol_ts.append({"date": str(d)[:10], "vol": round(float(v), 4)})

    return jsonify({
        "n_obs"          : len(returns),
        "mean_daily_ret" : round(float(returns.mean()), 6),
        "ann_volatility" : round(ann_vol, 4),
        "skewness"       : round(float(returns.skew()), 4),
        "excess_kurtosis": round(float(returns.kurtosis()), 4),
        "jb_p_value"     : round(float(jb_p), 6),
        "is_normal"      : bool(jb_p > 0.05),
        "histogram"      : hist,
        "rolling_vol"    : vol_ts[-252:],   # last year
    })


# --- Backtest summary (all tickers) ------------------------------------------

@app.route("/api/backtest-summary")
def get_backtest_summary():
    """Return the combined backtest_summary.csv."""
    path = OUTPUT_DIR / "backtest_summary.csv"
    df   = _safe_read_csv(path)
    if df.empty:
        return jsonify([])
    return jsonify(_jsonify_df(df.reset_index()))


# =============================================================================
# PIPELINE CONTROL
# =============================================================================

@app.route("/api/pipeline/status")
def pipeline_status():
    return jsonify({
        "running"   : pipeline_state["running"],
        "progress"  : pipeline_state["progress"],
        "log"       : pipeline_state["log"][-100:],   # last 100 lines
        "started_at": pipeline_state["started_at"],
    })


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """
    Trigger training pipeline in a background thread.
    POST body: { "tickers": ["RELIANCE.NS", "TCS.NS"], "lstm": false }
    """
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline already running"}), 409

    body    = request.get_json(force=True) or {}
    tickers = body.get("tickers", ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    lstm    = body.get("lstm", False)

    def _run():
        pipeline_state.update({"running": True, "log": [], "progress": 0,
                                "started_at": datetime.now().isoformat()})
        cmd = [sys.executable, str(BASE_DIR / "pipeline.py"),
               "--mode", "train",
               "--tickers"] + tickers
        if lstm:
            cmd.append("--lstm")

        steps = ["Data Collection", "EDA", "Feature Engineering",
                 "Random Forest", "XGBoost", "Ensemble", "Backtest"]
        step_idx = 0

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(BASE_DIR)
            )
            for line in proc.stdout:
                line = line.rstrip()
                pipeline_state["log"].append(line)
                # Advance progress bar by detecting step markers in log
                for i, step in enumerate(steps):
                    if step.upper() in line.upper() and i >= step_idx:
                        step_idx = i + 1
                        pipeline_state["progress"] = int(step_idx / len(steps) * 100)
            proc.wait()
            pipeline_state["progress"] = 100
        except Exception as e:
            pipeline_state["log"].append(f"ERROR: {e}")
        finally:
            pipeline_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "tickers": tickers})


@app.route("/api/pipeline/predict", methods=["POST"])
def run_predict():
    """Generate today's signals for given tickers."""
    body    = request.get_json(force=True) or {}
    tickers = body.get("tickers", _available_tickers())

    cmd = [sys.executable, str(BASE_DIR / "pipeline.py"),
           "--mode", "predict", "--tickers"] + tickers
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR))
        return jsonify({
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-500:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "timeout"}), 504


# =============================================================================
# SERVE REACT BUILD (production)
# =============================================================================

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serve the built React app for any non-API route."""
    build_dir = BASE_DIR / "dashboard" / "build"
    if build_dir.exists():
        if path and (build_dir / path).exists():
            return send_from_directory(str(build_dir), path)
        return send_from_directory(str(build_dir), "index.html")
    return jsonify({"message": "React build not found. Run: cd dashboard && npm run build"}), 200


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AlgoTrader API  ->  http://localhost:5000")
    print("  Endpoints:")
    print("    GET  /api/tickers")
    print("    GET  /api/price/<ticker>")
    print("    GET  /api/metrics/<ticker>")
    print("    GET  /api/backtest/<ticker>")
    print("    GET  /api/signals")
    print("    GET  /api/features/<ticker>")
    print("    GET  /api/eda/<ticker>")
    print("    GET  /api/backtest-summary")
    print("    POST /api/pipeline/run")
    print("    POST /api/pipeline/predict")
    print("    GET  /api/pipeline/status")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
