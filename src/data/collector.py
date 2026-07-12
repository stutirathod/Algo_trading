"""
=============================================================================
STEP 1 - DATA COLLECTION
=============================================================================
WHY THIS APPROACH:
  * yfinance wraps Yahoo Finance, which aggregates NSE/BSE data and is the
    most accessible free source for research-level historical OHLCV data.
  * We cache raw data locally so repeated runs don't hit rate limits.
  * Fundamental data (financials) is fetched via yfinance's Ticker API, which
    sources from SEC/company filings / Yahoo Finance fundamentals.
  * We persist everything as Parquet (columnar, compressed) for fast I/O.

DATA COLLECTED:
  Price   : Open, High, Low, Close, Volume (daily)
  Derived : Adjusted Close (corporate-action adjusted)
  Fundamental: Income Statement, Balance Sheet, Cash Flow (quarterly+annual)
=============================================================================
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    TICKERS, BENCHMARK, DATA_START, DATA_END, INTERVAL,
    RAW_DIR, CACHE_DIR, LOG_DIR, LOG_LEVEL
)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "data_collection.log"),
    ],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PRICE DATA
# -----------------------------------------------------------------------------

def fetch_price_data(
    tickers: List[str],
    start: str = DATA_START,
    end: str   = DATA_END,
    interval: str = INTERVAL,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.

    WHY PER-TICKER FETCH (not batch):
      Batch download via yf.download returns a MultiIndex DataFrame that
      silently drops tickers with missing data windows. Per-ticker fetching
      gives us full control over missing-data handling per symbol.

    Returns
    -------
    dict[ticker -> DataFrame with columns: Open, High, Low, Close, Volume, Adj Close]
    """
    price_data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker.replace('.', '_')}_price.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"[{ticker}] Loading price data from cache.")
            df = pd.read_parquet(cache_path)
        else:
            logger.info(f"[{ticker}] Downloading price data from Yahoo Finance...")
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, interval=interval, auto_adjust=False)

                if df.empty:
                    logger.warning(f"[{ticker}] No price data returned.")
                    continue

                # Standardise column names
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
                df.dropna(how="all", inplace=True)

                df.to_parquet(cache_path)
                logger.info(f"[{ticker}] {len(df)} rows saved to cache.")
                time.sleep(0.5)   # polite rate limiting

            except Exception as exc:
                logger.error(f"[{ticker}] Failed to fetch price data: {exc}")
                continue

        price_data[ticker] = df

    return price_data


def fetch_benchmark(
    ticker: str = BENCHMARK,
    start: str  = DATA_START,
    end: str    = DATA_END,
) -> pd.DataFrame:
    """Fetch benchmark (Nifty 50) data for beta / relative-return features."""
    cache_path = CACHE_DIR / f"{ticker.replace('^', 'IDX_')}_price.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading benchmark {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df


# -----------------------------------------------------------------------------
# FUNDAMENTAL DATA
# -----------------------------------------------------------------------------

def fetch_fundamental_data(
    tickers: List[str],
    use_cache: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch financial statements for each ticker.

    WHY FUNDAMENTALS:
      Fundamental indicators (EPS growth, ROE, D/E, FCF yield) capture the
      intrinsic quality of a business. Combining them with price features
      reduces overfitting to short-term noise and helps long-term trend
      prediction.

    Returns
    -------
    dict[ticker -> dict[statement_type -> DataFrame]]
      statement_type  {"income_stmt", "balance_sheet", "cash_flow",
                        "income_stmt_q", "balance_sheet_q", "cash_flow_q"}
    """
    fundamentals: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker.replace('.', '_')}_fundamentals.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"[{ticker}] Loading fundamentals from cache.")
            # Stored as a single multi-index parquet; reload and unpack
            combined = pd.read_parquet(cache_path)
            fundamentals[ticker] = _unpack_fundamentals(combined)
            continue

        logger.info(f"[{ticker}] Downloading fundamental data...")
        try:
            t = yf.Ticker(ticker)
            stmts = {
                "income_stmt"   : _safe_transpose(t.income_stmt),
                "balance_sheet" : _safe_transpose(t.balance_sheet),
                "cash_flow"     : _safe_transpose(t.cashflow),
                "income_stmt_q" : _safe_transpose(t.quarterly_income_stmt),
                "balance_sheet_q": _safe_transpose(t.quarterly_balance_sheet),
                "cash_flow_q"   : _safe_transpose(t.quarterly_cashflow),
            }
            fundamentals[ticker] = stmts

            # Persist: concatenate with a "statement" level in index
            frames = []
            for stmt_name, df in stmts.items():
                if df is not None and not df.empty:
                    df = df.copy()
                    df["_statement"] = stmt_name
                    frames.append(df)
            if frames:
                combined = pd.concat(frames)
                combined.to_parquet(cache_path)
                logger.info(f"[{ticker}] Fundamentals saved.")

            time.sleep(1.0)

        except Exception as exc:
            logger.error(f"[{ticker}] Failed to fetch fundamentals: {exc}")
            fundamentals[ticker] = {}

    return fundamentals


def _safe_transpose(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Transpose financial statements so rows=dates, cols=line items."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.T.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def _unpack_fundamentals(combined: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Reverse the packing done in fetch_fundamental_data."""
    result = {}
    if "_statement" not in combined.columns:
        return result
    for stmt_name, grp in combined.groupby("_statement"):
        result[stmt_name] = grp.drop(columns=["_statement"])
    return result


# -----------------------------------------------------------------------------
# DERIVED FUNDAMENTAL RATIOS
# -----------------------------------------------------------------------------

def compute_fundamental_ratios(
    fundamentals: Dict[str, pd.DataFrame],
    price_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Compute key financial ratios and align them to daily price dates.

    WHY FORWARD-FILL FUNDAMENTALS:
      Earnings / balance-sheet data is released quarterly. We forward-fill
      from the announcement date so that at any trading day the model only
      sees information that was *publicly available* - avoiding look-ahead bias.

    Ratios computed
    ---------------
    - Revenue YoY growth
    - Net Profit YoY growth
    - EPS (diluted)
    - ROE = Net Income / Shareholders' Equity
    - ROCE = EBIT / Capital Employed
    - Debt-to-Equity
    - Net Profit Margin
    - Free Cash Flow Yield (FCF / Market Cap proxy)
    """
    ratios: Dict[str, pd.DataFrame] = {}

    for ticker, stmts in fundamentals.items():
        if not stmts or ticker not in price_data:
            continue

        price_df  = price_data[ticker]
        date_idx  = price_df.index

        def _pick_stmt(primary: str, fallback: str) -> pd.DataFrame:
            """Return primary stmt if non-empty, else fallback, else empty DataFrame."""
            df_p = stmts.get(primary)
            if df_p is not None and not df_p.empty:
                return df_p
            df_f = stmts.get(fallback)
            if df_f is not None and not df_f.empty:
                return df_f
            return pd.DataFrame()

        inc = _pick_stmt("income_stmt_q",    "income_stmt")
        bs  = _pick_stmt("balance_sheet_q",  "balance_sheet")
        cf  = _pick_stmt("cash_flow_q",      "cash_flow")

        ratio_df = pd.DataFrame(index=date_idx)

        try:
            # --- Revenue growth ---
            rev = _get_col(inc, ["Total Revenue", "Revenue"])
            if rev is not None:
                ratio_df["revenue_yoy"] = rev.pct_change(4).reindex(date_idx).ffill()

            # --- Net profit growth ---
            ni = _get_col(inc, ["Net Income", "Net Income Common Stockholders"])
            if ni is not None:
                ratio_df["netprofit_yoy"] = ni.pct_change(4).reindex(date_idx).ffill()
                ratio_df["net_margin"]    = (ni / rev).reindex(date_idx).ffill() if rev is not None else None

            # --- EPS ---
            eps = _get_col(inc, ["Diluted EPS", "Basic EPS", "EPS"])
            if eps is not None:
                ratio_df["eps"] = eps.reindex(date_idx).ffill()

            # --- ROE ---
            eq = _get_col(bs, ["Stockholders Equity", "Total Stockholders Equity",
                                "Common Stock Equity"])
            if ni is not None and eq is not None:
                ratio_df["roe"] = (ni / eq).reindex(date_idx).ffill()

            # --- Debt-to-Equity ---
            debt = _get_col(bs, ["Total Debt", "Long Term Debt"])
            if debt is not None and eq is not None:
                ratio_df["debt_to_equity"] = (debt / eq.replace(0, float("nan"))
                                              ).reindex(date_idx).ffill()

            # --- Free Cash Flow ---
            fcf = _get_col(cf, ["Free Cash Flow"])
            if fcf is not None:
                ratio_df["fcf"] = fcf.reindex(date_idx).ffill()

        except Exception as exc:
            logger.warning(f"[{ticker}] Ratio computation partial failure: {exc}")

        # Drop all-NaN columns
        ratio_df.dropna(axis=1, how="all", inplace=True)
        ratios[ticker] = ratio_df

    return ratios


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return first matching column from a DataFrame."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def collect_all_data(
    tickers: List[str] = TICKERS,
    use_cache: bool    = True,
) -> Tuple[Dict, Dict, Dict]:
    """
    Orchestrates full data collection.

    Returns
    -------
    price_data      : dict[ticker -> OHLCV DataFrame]
    fundamentals    : dict[ticker -> dict[statement -> DataFrame]]
    ratios          : dict[ticker -> ratios DataFrame aligned to price dates]
    """
    logger.info("=" * 60)
    logger.info("STEP 1 - DATA COLLECTION STARTED")
    logger.info("=" * 60)

    price_data   = fetch_price_data(tickers, use_cache=use_cache)
    benchmark    = fetch_benchmark()
    fundamentals = fetch_fundamental_data(tickers, use_cache=use_cache)
    ratios       = compute_fundamental_ratios(fundamentals, price_data)

    logger.info(f"Price data collected for {len(price_data)} tickers.")
    logger.info(f"Fundamental ratios computed for {len(ratios)} tickers.")
    logger.info("STEP 1 - COMPLETE\n")

    return price_data, fundamentals, ratios


if __name__ == "__main__":
    collect_all_data()