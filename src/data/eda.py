"""
=============================================================================
STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
PURPOSE:
  Before choosing any model, we must *understand the data*. The EDA answers:
    1. What is the return distribution? (Normal? Fat-tailed? Skewed?)
    2. Are there missing values / data gaps?
    3. Are there outliers that could destabilise training?
    4. Which features correlate with future returns?
    5. Is the price series stationary? (crucial for time-series models)
    6. What volatility regime does the data exhibit?
    7. Are patterns linear or nonlinear?

  Each finding directly informs model selection in Step 4.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import OUTPUT_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. MISSING VALUE ANALYSIS
# -----------------------------------------------------------------------------

def analyse_missing(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    WHY: Missing OHLCV bars (market holidays, halts) must be handled before
    feature engineering or they propagate NaN through rolling calculations.
    """
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    report  = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report  = report[report["missing_count"] > 0]

    if not report.empty:
        logger.warning(f"[{ticker}] Missing values found:\n{report}")
    else:
        logger.info(f"[{ticker}] No missing values.")

    return report


# -----------------------------------------------------------------------------
# 2. RETURN DISTRIBUTION
# -----------------------------------------------------------------------------

def analyse_return_distribution(
    price_df: pd.DataFrame,
    ticker: str = "",
    save: bool  = True,
) -> Dict:
    """
    WHY: If returns are normally distributed, linear models (OLS/ARIMA) are
    sufficient. Fat tails / skew -> need robust nonlinear models (XGBoost,
    LSTM) and careful risk sizing.

    Tests performed:
      - Jarque-Bera (normality)
      - Skewness / excess kurtosis
      - Annualised volatility
    """
    close   = price_df["Close"].dropna()
    returns = close.pct_change().dropna()

    jb_stat, jb_p = stats.jarque_bera(returns)
    skew   = float(returns.skew())
    kurt   = float(returns.kurtosis())   # excess kurtosis
    ann_vol = float(returns.std() * np.sqrt(252))

    findings = {
        "ticker"         : ticker,
        "n_obs"          : len(returns),
        "mean_daily_ret" : float(returns.mean()),
        "ann_volatility" : ann_vol,
        "skewness"       : skew,
        "excess_kurtosis": kurt,
        "jb_stat"        : jb_stat,
        "jb_p_value"     : jb_p,
        "is_normal"      : jb_p > 0.05,
    }

    # --- Interpretation -------------------------------------------------------
    interpretation = []
    if not findings["is_normal"]:
        interpretation.append(
            "Returns are NOT normally distributed (Jarque-Bera p < 0.05). "
            "Fat tails / skew are present -> linear models will underestimate tail risk."
        )
    if abs(kurt) > 1:
        interpretation.append(
            f"Excess kurtosis = {kurt:.2f} -> fat tails. "
            "XGBoost / LSTM handle this better than OLS."
        )
    if abs(skew) > 0.5:
        interpretation.append(
            f"Significant skew = {skew:.2f}. "
            "Asymmetric loss functions or direction-based targets recommended."
        )
    findings["interpretation"] = interpretation
    for line in interpretation:
        logger.info(f"[{ticker}] {line}")

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(returns, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].set_title(f"{ticker} - Daily Return Distribution")
        axes[0].set_xlabel("Daily Return")
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title(f"{ticker} - Q-Q Plot (vs Normal)")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{ticker}_return_dist.png", dpi=120)
        plt.close()

    return findings


# -----------------------------------------------------------------------------
# 3. OUTLIER DETECTION
# -----------------------------------------------------------------------------

def detect_outliers(
    price_df: pd.DataFrame,
    ticker: str = "",
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    WHY: Extreme daily moves (>4sigma) are often data errors or black-swan events.
    We flag them so we can decide whether to winsorise or drop them before
    training (unhandled outliers cause gradient explosions in neural networks).
    """
    returns = price_df["Close"].pct_change().dropna()
    z_scores = np.abs(stats.zscore(returns))
    outliers = returns[z_scores > z_threshold]

    logger.info(f"[{ticker}] Outliers (|z|>{z_threshold}): {len(outliers)} days")
    if not outliers.empty:
        logger.info(f"[{ticker}] Dates: {outliers.index.tolist()}")

    return outliers.to_frame(name="outlier_return")


# -----------------------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# -----------------------------------------------------------------------------

def analyse_correlations(
    feature_df: pd.DataFrame,
    target_col: str = "target",
    ticker: str     = "",
    top_n: int      = 20,
    save: bool      = True,
) -> pd.Series:
    """
    WHY: Correlation analysis reveals which features have *linear* predictive
    power. Low correlations don't mean a feature is useless (it may have
    *nonlinear* predictive power for tree models), but very high inter-feature
    correlations flag multicollinearity that can degrade linear models.
    """
    if target_col not in feature_df.columns:
        logger.warning(f"[{ticker}] Target column '{target_col}' not found.")
        return pd.Series()

    corr_with_target = (
        feature_df.select_dtypes(include=[np.number])
        .corr()[target_col]
        .drop(target_col)
        .dropna()
        .sort_values(key=abs, ascending=False)
    )

    top = corr_with_target.head(top_n)
    logger.info(f"[{ticker}] Top {top_n} features by |corr| with target:\n{top.round(4)}")

    if save:
        # Feature-feature heatmap for top correlated features
        top_features = top.index.tolist() + [target_col]
        sub = feature_df[top_features].dropna()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(sub.corr(), dtype=bool))
        sns.heatmap(sub.corr(), mask=mask, annot=False, cmap="RdBu_r",
                    center=0, ax=ax, linewidths=0.3)
        ax.set_title(f"{ticker} - Feature Correlation Heatmap (top {top_n})")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{ticker}_corr_heatmap.png", dpi=120)
        plt.close()

    return corr_with_target


# -----------------------------------------------------------------------------
# 5. STATIONARITY TESTS
# -----------------------------------------------------------------------------

def test_stationarity(
    series: pd.Series,
    name: str   = "series",
    ticker: str = "",
) -> Dict:
    """
    WHY THIS MATTERS FOR MODEL SELECTION:
      - Raw price levels are non-stationary (unit root process). Feeding raw
        prices into a vanilla LSTM will cause it to learn the *trend*, not
        the *signal* -> overfits to the training period's price level.
      - Returns / log-returns are typically stationary -> preferred features.
      - ADF test: H0 = unit root (non-stationary). Reject H0 -> stationary.
      - KPSS test: H0 = stationary. Reject H0 -> non-stationary.
      - If ADF rejects AND KPSS does NOT reject -> strong evidence of stationarity.
    """
    clean = series.dropna()

    # ADF
    adf_result = adfuller(clean, autolag="AIC")
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adf_result

    # KPSS
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(clean, regression="c", nlags="auto")
        kpss_stationary = kpss_p > 0.05   # don't reject H0
    except Exception:
        kpss_stat = kpss_p = None
        kpss_stationary = None

    result = {
        "name"            : name,
        "ticker"          : ticker,
        "adf_statistic"   : adf_stat,
        "adf_p_value"     : adf_p,
        "adf_stationary"  : adf_p < 0.05,
        "kpss_statistic"  : kpss_stat,
        "kpss_p_value"    : kpss_p,
        "kpss_stationary" : kpss_stationary,
    }

    verdict = (
        "STATIONARY" if (result["adf_stationary"] and kpss_stationary)
        else "NON-STATIONARY" if (not result["adf_stationary"] and not kpss_stationary)
        else "AMBIGUOUS"
    )
    result["verdict"] = verdict
    logger.info(f"[{ticker}] Stationarity of '{name}': {verdict} "
                f"(ADF p={adf_p:.4f}, KPSS p={kpss_p})")

    return result


# -----------------------------------------------------------------------------
# 6. VOLATILITY ANALYSIS (ARCH EFFECTS)
# -----------------------------------------------------------------------------

def analyse_volatility(
    price_df: pd.DataFrame,
    ticker: str = "",
    save: bool  = True,
) -> Dict:
    """
    WHY: ARCH / GARCH effects (volatility clustering) are a hallmark of
    equity returns. If present, features capturing *current volatility regime*
    (e.g., realised vol, ATR, Bollinger Band width) become very informative.
    We use the ARCH-LM test to verify clustering before engineering vol features.
    """
    returns = price_df["Close"].pct_change().dropna()

    # ARCH LM test
    try:
        lm_stat, lm_p, f_stat, f_p = het_arch(returns, nlags=10)
        arch_present = lm_p < 0.05
    except Exception:
        lm_stat = lm_p = None
        arch_present = None

    # Rolling 20-day realised volatility
    roll_vol = returns.rolling(20).std() * np.sqrt(252)

    result = {
        "arch_lm_stat"    : lm_stat,
        "arch_lm_p"       : lm_p,
        "arch_present"    : arch_present,
        "mean_ann_vol"    : float(roll_vol.mean()),
        "max_ann_vol"     : float(roll_vol.max()),
        "vol_vol"         : float(roll_vol.std()),   # volatility of volatility
    }
    logger.info(f"[{ticker}] ARCH effects present: {arch_present} (p={lm_p:.4f})")

    if save:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(price_df["Close"], color="steelblue", lw=0.8)
        axes[0].set_title(f"{ticker} - Price")
        axes[1].plot(roll_vol, color="darkorange", lw=0.8)
        axes[1].set_title(f"{ticker} - Rolling 20-day Annualised Volatility")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{ticker}_volatility.png", dpi=120)
        plt.close()

    return result


# -----------------------------------------------------------------------------
# 7. AUTOCORRELATION ANALYSIS
# -----------------------------------------------------------------------------

def analyse_autocorrelation(
    price_df: pd.DataFrame,
    ticker: str = "",
    lags: int   = 40,
    save: bool  = True,
) -> None:
    """
    WHY: Significant autocorrelation in returns -> momentum / mean-reversion
    is exploitable. This informs whether lag features and LSTM lookback
    windows are useful.
    """
    returns = price_df["Close"].pct_change().dropna()

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(returns, lags=lags, ax=axes[0], title=f"{ticker} - ACF of Returns")
        plot_pacf(returns, lags=lags, ax=axes[1],
                  title=f"{ticker} - PACF of Returns", method="ywm")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{ticker}_autocorr.png", dpi=120)
        plt.close()

    logger.info(f"[{ticker}] ACF/PACF plots saved.")


# -----------------------------------------------------------------------------
# 8. NONLINEARITY CHECK
# -----------------------------------------------------------------------------

def check_nonlinearity(
    feature_df: pd.DataFrame,
    target_col: str = "target",
    ticker: str     = "",
    top_features: int = 6,
    save: bool      = True,
) -> None:
    """
    WHY: If the relationship between features and target is nonlinear, tree
    models (RF, XGBoost) and neural networks will outperform linear regression.
    We visualise scatter plots with a LOWESS smoother.
    """
    if target_col not in feature_df.columns or save is False:
        return

    corr = (feature_df.select_dtypes(include=[np.number])
            .corr()[target_col].drop(target_col).abs()
            .sort_values(ascending=False))
    features = corr.head(top_features).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, feat in zip(axes.flatten(), features):
        sub = feature_df[[feat, target_col]].dropna()
        ax.scatter(sub[feat], sub[target_col], alpha=0.2, s=5, color="steelblue")
        # LOWESS smoother
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(sub[target_col], sub[feat], frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color="red", lw=2, label="LOWESS")
        ax.set_xlabel(feat)
        ax.set_ylabel(target_col)
        ax.set_title(feat)
        ax.legend(fontsize=7)
    plt.suptitle(f"{ticker} - Feature vs Target (nonlinearity check)", y=1.02)
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{ticker}_nonlinearity.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"[{ticker}] Nonlinearity plots saved.")


# -----------------------------------------------------------------------------
# MASTER EDA FUNCTION
# -----------------------------------------------------------------------------

def run_eda(
    price_data: Dict[str, pd.DataFrame],
    feature_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Dict]:
    """
    Run the full EDA pipeline for all tickers and compile model-selection
    recommendations.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 - EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    all_findings: Dict[str, Dict] = {}

    for ticker, price_df in price_data.items():
        logger.info(f"\n{'-'*40}\nAnalysing {ticker}\n{'-'*40}")
        findings = {}

        findings["missing"]       = analyse_missing(price_df, ticker)
        findings["distribution"]  = analyse_return_distribution(price_df, ticker)
        findings["outliers"]      = detect_outliers(price_df, ticker)

        close     = price_df["Close"].dropna()
        returns   = close.pct_change().dropna()
        log_ret   = np.log(close / close.shift(1)).dropna()

        findings["stationarity_price"]  = test_stationarity(close,   "price",   ticker)
        findings["stationarity_return"] = test_stationarity(returns, "returns", ticker)
        findings["stationarity_logret"] = test_stationarity(log_ret, "log_ret", ticker)
        findings["volatility"]          = analyse_volatility(price_df, ticker)
        analyse_autocorrelation(price_df, ticker)

        if feature_data and ticker in feature_data:
            findings["correlations"] = analyse_correlations(
                feature_data[ticker], "target", ticker
            )
            check_nonlinearity(feature_data[ticker], "target", ticker)

        # --- Model selection recommendation -------------------------------
        recommendation = _build_recommendation(findings)
        findings["recommendation"] = recommendation
        logger.info(f"[{ticker}] Model recommendation: {recommendation}")

        all_findings[ticker] = findings

    logger.info("STEP 2 - EDA COMPLETE\n")
    return all_findings


def _build_recommendation(findings: Dict) -> str:
    """
    Translate EDA findings into model-selection reasoning.
    """
    dist   = findings.get("distribution", {})
    vol    = findings.get("volatility", {})
    stat_r = findings.get("stationarity_return", {})

    notes = []

    if not dist.get("is_normal", True):
        notes.append(
            "Non-normal returns -> LSTM + XGBoost preferred over OLS/ARIMA."
        )
    if vol.get("arch_present"):
        notes.append(
            "ARCH effects detected -> volatility features (ATR, BB width, realised vol) "
            "are informative. LSTM can learn regime changes."
        )
    if stat_r.get("adf_stationary"):
        notes.append(
            "Returns are stationary -> suitable as features/targets directly. "
            "No differencing needed."
        )
    if not findings.get("stationarity_price", {}).get("adf_stationary", True):
        notes.append(
            "Price levels are non-stationary -> must use returns / log-returns as "
            "features, not raw price levels."
        )

    notes.append(
        "Strong temporal dependency present -> LSTM for sequence modelling. "
        "XGBoost/RF for feature-interaction capture. Ensemble to combine."
    )

    return " | ".join(notes)
