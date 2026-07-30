"""
data_manager.py  —  HuggingFace data loader for the Free Energy-Principled Agent
=================================================================================

Loads the master parquet from P2SAMAPA/fi-etf-macro-signal-master-data,
extracts ETF price columns and macro signal columns.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

import config

logger = logging.getLogger(__name__)

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
MACRO_COLS_ALL = [col for col, _, _, _ in config.MACRO_SIGNALS]


def _all_tickers() -> List[str]:
    """Deduplicated list of every ticker across all universes."""
    seen, result = set(), []
    for tickers in config.UNIVERSES.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def load_master_data(
    hf_token: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download master parquet from HuggingFace and return (prices, macro).
    """
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN not set — cannot load data from HuggingFace.")

    api = HfApi(token=token)

    logger.info(f"Downloading master parquet from {config.DATA_REPO} …")
    parquet_path = None
    for fname in [
        "master_data.parquet",
        "data/master.parquet",
        "master.parquet",
        "fi_etf_macro_master.parquet",
        "dataset.parquet",
    ]:
        try:
            parquet_path = api.hf_hub_download(
                repo_id=config.DATA_REPO,
                filename=fname,
                repo_type="dataset",
                token=token,
            )
            logger.info(f"  → found at '{fname}'")
            break
        except Exception:
            continue

    if parquet_path is None:
        raise RuntimeError(
            f"Could not locate master parquet in {config.DATA_REPO}. "
            "Check the filename and repo layout."
        )

    df = pd.read_parquet(parquet_path)
    logger.info(f"Raw parquet: {df.shape[0]} rows × {df.shape[1]} cols")

    if not isinstance(df.index, pd.DatetimeIndex):
        for date_col in ["Date", "date", "DATE", "timestamp", "Timestamp"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df.index.name = "Date"

    all_tickers = _all_tickers()
    avail_tickers = [t for t in all_tickers if t in df.columns]
    missing_tick = [t for t in all_tickers if t not in df.columns]

    if missing_tick:
        logger.warning(f"Tickers not found in parquet: {missing_tick}")
    if not avail_tickers:
        raise RuntimeError("No ETF ticker columns found in master parquet.")

    prices = df[avail_tickers].copy()
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    avail_core = [c for c in MACRO_COLS_CORE if c in df.columns]
    avail_ext = [c for c in MACRO_COLS_EXTENDED if c in df.columns]
    avail_all = avail_core + avail_ext
    missing_mac = [c for c in MACRO_COLS_ALL if c not in df.columns]

    if missing_mac:
        logger.warning(f"Macro columns not found in parquet: {missing_mac}")
    if not avail_all:
        raise RuntimeError("No macro columns found in master parquet.")

    macro = df[avail_all].copy()

    if avail_core:
        before = len(macro)
        macro = macro.dropna(subset=avail_core)
        dropped = before - len(macro)
        if dropped:
            logger.info(f"Dropped {dropped} rows with NaN in core macro cols.")

    if avail_ext:
        macro[avail_ext] = macro[avail_ext].ffill().fillna(0.0)

    common = prices.index.intersection(macro.index)
    if len(common) == 0:
        raise RuntimeError("No overlapping dates between price and macro data.")

    prices = prices.loc[common]
    macro = macro.loc[common]

    logger.info(
        f"Dataset ready: {len(prices)} rows | "
        f"{len(avail_tickers)} ETFs | "
        f"{len(avail_all)} macro cols | "
        f"{prices.index[0].date()} → {prices.index[-1].date()}"
    )

    return prices, macro


def validate_data(prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    """Sanity-check loaded data."""
    for ticker in list(prices.columns)[:3]:
        col = prices[ticker].dropna()
        if len(col) > 10 and abs(col.median()) < 0.05:
            logger.warning(
                f"'{ticker}' median ≈ {col.median():.4f} — looks like returns, "
                "not prices! Log-return computation will be wrong."
            )

    nan_counts = macro.isnull().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        logger.warning(f"Residual macro NaN counts after cleaning:\n{bad}")

    if len(prices) < 252:
        logger.warning(
            f"Only {len(prices)} rows — less than 1 year of data. "
            "Longer windows will return NaN."
        )
