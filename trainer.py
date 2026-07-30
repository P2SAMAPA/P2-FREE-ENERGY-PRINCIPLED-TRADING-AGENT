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

    # ── Collect all signals across all universes for global z-score ──────────
    all_signals = {}  # ticker -> signal_value

    # ── Store per-ticker results across all universes ────────────────────────
    all_ticker_data = {}  # ticker -> {action, free_energy, surprise, epistemic, position, window_signals}

    # ── Process each universe ─────────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n🧠 Processing universe: {universe_name}")

        # Filter available tickers
        available = [t for t in tickers if t in prices_df.columns]
        logger.info(f"   Available: {len(available)}/{len(tickers)}")

        if not available:
            logger.warning(f"   No tickers available for {universe_name}")
            continue

        # Store results for this universe
        universe_ticker_data = {}

        # ── Compute for each ticker ────────────────────────────────────────────
        for ticker in available:
            logger.info(f"   Computing {ticker}...")
            prices = prices_df[ticker]
            
            try:
                result = compute_agent_signal(prices, macro_df, agent_config)
                
                # Check for errors
                error = result.get("error")
                if error is not None and str(error) != "None" and str(error) != "":
                    logger.warning(f"      {ticker}: Error - {error}")
                    continue
                
                # Get values
                action = result.get("action", "HOLD")
                if action == "INSUFFICIENT DATA":
                    logger.warning(f"      {ticker}: Insufficient data")
                    continue
                
                free_energy = safe_float(result.get("free_energy", 0))
                surprise = safe_float(result.get("surprise", 0))
                epistemic = safe_float(result.get("epistemic", 0))
                position = safe_float(result.get("position", 0))
                window_signals = result.get("window_signals", [])
                
                # Map action to numeric for scoring
                action_map = {"STRONG BUY": 1.5, "BUY": 1.0, "HOLD": 0.0, "REDUCE": -0.5, "STRONG SELL": -1.0}
                signal_value = action_map.get(action, 0.0)
                
                # Store in all signals for global z-score
                all_signals[ticker] = signal_value
                
                # Store full data
                all_ticker_data[ticker] = {
                    "action": action,
                    "free_energy": free_energy,
                    "surprise": surprise,
                    "epistemic": epistemic,
                    "position": position,
                    "window_signals": window_signals,
                    "universe": universe_name,
                    "signal": signal_value
                }
                
                universe_ticker_data[ticker] = {
                    "action": action,
                    "free_energy": free_energy,
                    "surprise": surprise,
                    "epistemic": epistemic,
                    "position": position,
                    "window_signals": window_signals,
                    "signal": signal_value
                }
                
                logger.info(f"      {ticker}: {action} | F={free_energy:.3f} | Signal={signal_value:.2f}")
                
            except Exception as e:
                logger.error(f"      {ticker}: Exception - {str(e)}")
                continue

    # ── Compute GLOBAL z-scores across all tickers ──────────────────────────
    logger.info("\n📊 Computing global z-scores across all tickers...")
    global_z_scores = compute_cross_sectional_zscore(all_signals)
    
    # ── Build results by universe ────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        # Get tickers in this universe that have data
        universe_tickers = [t for t in tickers if t in all_ticker_data]
        
        if not universe_tickers:
            continue
        
        # Build full scores with global z-scores
        full_scores = {}
        signal_values = {}
        
        for ticker in universe_tickers:
            data = all_ticker_data[ticker]
            z_score = global_z_scores.get(ticker, 0.0)
            signal_values[ticker] = z_score
            
            full_scores[ticker] = {
                "z_score": z_score,
                "action": data["action"],
                "free_energy": data["free_energy"],
                "surprise": data["surprise"],
                "epistemic": data["epistemic"],
                "position": data["position"],
                "signal": data["signal"],
                "window_signals": data.get("window_signals", [])
            }
        
        # Top 5 buys (highest z-score)
        top_buys = sorted(
            [(t, full_scores[t]["z_score"]) for t in full_scores if not np.isnan(full_scores[t]["z_score"])],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Top 5 sells (lowest z-score)
        top_sells = sorted(
            [(t, full_scores[t]["z_score"]) for t in full_scores if not np.isnan(full_scores[t]["z_score"])],
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
            "full_scores": full_scores
        }
        
        # Build Tab 2
        results_tab2["universes"][universe_name] = {
            "full_ranking": [
                {
                    "ticker": t,
                    "z_score": full_scores[t]["z_score"],
                    "action": full_scores[t]["action"],
                    "free_energy": full_scores[t]["free_energy"],
                    "surprise": full_scores[t]["surprise"],
                    "epistemic": full_scores[t]["epistemic"],
                    "position": full_scores[t]["position"],
                    "signal": full_scores[t]["signal"],
                    "window_signals": full_scores[t].get("window_signals", [])
                }
                for t in full_scores.keys()
            ]
        }
        
        logger.info(f"   ✅ {universe_name}: {len(full_scores)} tickers processed")

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


def safe_float(val, default=0.0):
    """Safely convert to float."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


if __name__ == "__main__":
    run_trainer()
