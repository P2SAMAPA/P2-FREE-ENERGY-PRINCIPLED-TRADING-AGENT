"""
trainer.py  —  Orchestrator for Free Energy-Principled Agent
=============================================================

Loads data → trains agent → generates FEP signals → builds JSON → uploads.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

import config
from data_manager import load_master_data, validate_data
from fep_agent import compute_agent_signal, compute_cross_sectional_zscore
from push_results import upload_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """
    Run the full Free Energy-Principled Agent pipeline.
    """
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(
        f"✅ Loaded {len(prices_df)} days, "
        f"{len(prices_df.columns)} ETFs, "
        f"{len(macro_df.columns)} macro cols"
    )

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Agent configuration ──────────────────────────────────────────────────
    agent_config = {
        **config.GENERATIVE_MODEL,
        **config.FREE_ENERGY,
        **config.ACTION_SPACE,
        **config.EPISTEMIC_DRIVE,
        "windows": config.WINDOWS,
        "primary_window": config.PRIMARY_WINDOW,
    }

    # ── Results containers ────────────────────────────────────────────────────
    results_tab1 = {
        "run_date": run_date,
        "universes": {}
    }

    results_tab2 = {
        "run_date": run_date,
        "universes": {}
    }

    # ── Process each universe ─────────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n🧠 Processing universe: {universe_name}")

        # Filter available tickers
        available = [t for t in tickers if t in prices_df.columns]
        logger.info(f"   Available: {len(available)}/{len(tickers)}")

        if not available:
            logger.warning(f"   No tickers available for {universe_name}")
            continue

        # Store results
        ticker_actions = {}
        ticker_free_energy = {}
        ticker_surprise = {}
        ticker_epistemic = {}
        ticker_position = {}
        ticker_signal = {}
        ticker_window_signals = {}

        # ── Compute for each ticker ────────────────────────────────────────────
        for ticker in available:
            logger.info(f"   Computing {ticker}...")
            prices = prices_df[ticker]
            
            try:
                result = compute_agent_signal(prices, macro_df, agent_config)
                
                if "error" not in result:
                    # Map action to numeric for scoring
                    action_map = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
                    signal_value = action_map.get(result.get("action", "HOLD"), 0.0)
                    
                    ticker_actions[ticker] = result.get("action", "HOLD")
                    ticker_free_energy[ticker] = result.get("free_energy", 0)
                    ticker_surprise[ticker] = result.get("surprise", 0)
                    ticker_epistemic[ticker] = result.get("epistemic", 0)
                    ticker_position[ticker] = result.get("position", 0)
                    ticker_signal[ticker] = signal_value
                    ticker_window_signals[ticker] = result.get("window_signals", [])
                    
                    logger.info(f"      {ticker}: {result.get('action', 'HOLD')} | F={result.get('free_energy', 0):.3f}")
                else:
                    logger.warning(f"      {ticker}: Error - {result.get('error')}")
            except Exception as e:
                logger.error(f"      {ticker}: Exception - {str(e)}")

        # ── Cross-sectional z-scores ──────────────────────────────────────────
        if ticker_signal:
            z_scores = compute_cross_sectional_zscore(ticker_signal)
            
            # Determine action based on z-score
            actions = {}
            for ticker, z in z_scores.items():
                if z > 0.5:
                    actions[ticker] = "STRONG BUY"
                elif z > 0.2:
                    actions[ticker] = "BUY"
                elif z > -0.2:
                    actions[ticker] = "HOLD"
                elif z > -0.5:
                    actions[ticker] = "REDUCE"
                else:
                    actions[ticker] = "STRONG SELL"
            
            # Top 5 buys (highest z-score)
            top_buys = sorted(
                [(t, z_scores[t]) for t in z_scores if not np.isnan(z_scores[t])],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Top 5 sells (lowest z-score)
            top_sells = sorted(
                [(t, z_scores[t]) for t in z_scores if not np.isnan(z_scores[t])],
                key=lambda x: x[1]
            )[:5]
            
            # Build Tab 1
            results_tab1["universes"][universe_name] = {
                "top_buys": [
                    {"ticker": t, "z_score": z} for t, z in top_buys
                ],
                "top_sells": [
                    {"ticker": t, "z_score": z} for t, z in top_sells
                ],
                "full_scores": {
                    t: {
                        "z_score": z_scores.get(t, 0),
                        "action": actions.get(t, "HOLD"),
                        "free_energy": ticker_free_energy.get(t, 0),
                        "surprise": ticker_surprise.get(t, 0),
                        "epistemic": ticker_epistemic.get(t, 0),
                        "position": ticker_position.get(t, 0),
                        "signal": ticker_signal.get(t, 0),
                        "window_signals": ticker_window_signals.get(t, [])
                    }
                    for t in ticker_signal.keys()
                }
            }
            
            # Build Tab 2
            results_tab2["universes"][universe_name] = {
                "full_ranking": [
                    {
                        "ticker": t,
                        "z_score": z_scores.get(t, 0),
                        "action": actions.get(t, "HOLD"),
                        "free_energy": ticker_free_energy.get(t, 0),
                        "surprise": ticker_surprise.get(t, 0),
                        "epistemic": ticker_epistemic.get(t, 0),
                        "position": ticker_position.get(t, 0),
                        "signal": ticker_signal.get(t, 0),
                        "window_signals": ticker_window_signals.get(t, [])
                    }
                    for t in ticker_signal.keys()
                ]
            }
            
            logger.info(f"   ✅ {universe_name}: {len(ticker_signal)} tickers processed")
        else:
            logger.warning(f"   ⚠️ No signals generated for {universe_name}")

    # ── Save JSON files ──────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")

    tab1_path = f"fep_agent_{run_date}.json"
    tab2_path = f"fep_agent_breakdown_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2, default=str)

    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2, default=str)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    # ── Upload to HuggingFace ───────────────────────────────────────────────
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            upload_results(tab1_path, tab2_path, token)
        except Exception as e:
            logger.error(f"   Upload failed: {e}")
    else:
        logger.info("\n📤 Skipping upload (no HF_TOKEN)")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
