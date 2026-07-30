"""
fep_agent.py  —  Free Energy-Principled Trading Agent (Simplified)
=======================================================

Uses statistical moments (mean, variance, skewness) to compute:
- Surprise: Deviation from expected returns
- Free Energy: Combined measure of risk and uncertainty
- Action selection: Choose actions that minimize expected free energy
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def compute_returns_features(returns: np.ndarray, window: int) -> Dict:
    """
    Compute statistical features from returns for a given window.
    """
    if len(returns) < window:
        window = len(returns)
    
    recent = returns[-window:]
    
    # Basic statistics
    mean = np.mean(recent)
    std = np.std(recent)
    skew = pd.Series(recent).skew()
    kurt = pd.Series(recent).kurtosis()
    
    # Rolling statistics
    rolling_mean = np.mean(recent[-20:]) if len(recent) >= 20 else mean
    rolling_std = np.std(recent[-20:]) if len(recent) >= 20 else std
    
    # Recent momentum
    short_term = np.mean(recent[-10:]) if len(recent) >= 10 else mean
    long_term = np.mean(recent[-min(60, len(recent)):]) if len(recent) >= 60 else mean
    
    # Max drawdown in window
    cum_returns = np.cumsum(recent)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Volatility ratio (short vs long)
    short_vol = np.std(recent[-10:]) if len(recent) >= 10 else std
    long_vol = np.std(recent[-min(60, len(recent)):]) if len(recent) >= 60 else std
    vol_ratio = short_vol / (long_vol + 1e-6)
    
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "short_term": short_term,
        "long_term": long_term,
        "max_drawdown": max_drawdown,
        "vol_ratio": vol_ratio,
        "n": len(recent)
    }


def compute_surprise(returns: np.ndarray, window: int = 252) -> float:
    """
    Compute surprise as the deviation from expected returns.
    Lower surprise = more predictable = better.
    """
    if len(returns) < 20:
        return 1.0
    
    features = compute_returns_features(returns, window)
    
    # Expected return based on long-term mean
    expected = features["long_term"]
    
    # Actual recent return
    actual = features["short_term"]
    
    # Surprise = absolute deviation from expectation
    surprise = abs(actual - expected) / (features["std"] + 1e-6)
    
    # Normalize to 0-10 range
    return min(10, surprise)


def compute_epistemic_value(returns: np.ndarray, window: int = 252) -> float:
    """
    Compute epistemic value (model uncertainty).
    Higher uncertainty = more exploration needed.
    """
    if len(returns) < 20:
        return 1.0
    
    features = compute_returns_features(returns, window)
    
    # Uncertainty measures:
    # 1. High volatility = high uncertainty
    # 2. High kurtosis = high tail uncertainty
    # 3. High vol_ratio = regime uncertainty
    
    vol_uncertainty = features["std"] / (np.mean(features["std"]) + 1e-6)
    tail_uncertainty = abs(features["kurt"]) / 3  # Normalize
    regime_uncertainty = abs(features["vol_ratio"] - 1) * 2
    
    epistemic = (vol_uncertainty + tail_uncertainty + regime_uncertainty) / 3
    return min(1.0, epistemic)


def compute_free_energy(returns: np.ndarray, window: int = 252) -> Dict:
    """
    Compute free energy components for a given window.
    """
    features = compute_returns_features(returns, window)
    
    # Surprise: how unexpected is the recent return?
    surprise = compute_surprise(returns, window)
    
    # Epistemic: how uncertain is the model?
    epistemic = compute_epistemic_value(returns, window)
    
    # Pragmatic value: inverse of surprise (lower surprise = higher pragmatic value)
    pragmatic_value = -surprise
    
    # Free energy = pragmatic + epistemic - KL (simplified)
    # Lower free energy = better state
    free_energy = pragmatic_value + epistemic * 0.5
    
    return {
        "free_energy": free_energy,
        "surprise": surprise,
        "epistemic": epistemic,
        "pragmatic_value": pragmatic_value,
        "mean": features["mean"],
        "std": features["std"],
        "skew": features["skew"],
        "kurt": features["kurt"],
        "max_drawdown": features["max_drawdown"],
        "vol_ratio": features["vol_ratio"]
    }


def compute_fep_signals(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    train_agent: bool = True
) -> Dict:
    """
    Compute Free Energy-Principled signals for a single ticker.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < 50:
        return {
            "action": "HOLD",
            "free_energy": 0.0,
            "surprise": 0.0,
            "epistemic": 0.0,
            "position": 0.0,
            "action_probabilities": [0.33, 0.33, 0.34],
            "window_signals": [],
            "error": "Insufficient data"
        }
    
    try:
        # Get windows from config
        windows = config.get("windows", [63, 252, 504, 1008, 2016, 4032])
        primary_window = config.get("primary_window", 252)
        
        # Compute signals for each window
        window_signals = []
        fe_values = []
        
        for window in windows:
            fe_result = compute_free_energy(returns, min(window, len(returns) - 1))
            window_signals.append({
                "window": window,
                "free_energy": fe_result["free_energy"],
                "surprise": fe_result["surprise"],
                "epistemic": fe_result["epistemic"],
                "pragmatic_value": fe_result["pragmatic_value"],
                "mean": fe_result["mean"],
                "std": fe_result["std"],
                "skew": fe_result["skew"],
                "kurt": fe_result["kurt"],
                "max_drawdown": fe_result["max_drawdown"]
            })
            fe_values.append(fe_result["free_energy"])
        
        # Weighted aggregate (primary window gets 40%)
        weights = {}
        for w in windows:
            if w == primary_window:
                weights[w] = 0.4
            else:
                weights[w] = 0.6 / (len(windows) - 1) if len(windows) > 1 else 0.6
        
        # Compute aggregate free energy
        agg_free_energy = 0
        agg_surprise = 0
        agg_epistemic = 0
        for ws in window_signals:
            w = ws["window"]
            agg_free_energy += ws["free_energy"] * weights[w]
            agg_surprise += ws["surprise"] * weights[w]
            agg_epistemic += ws["epistemic"] * weights[w]
        
        # Determine action based on free energy
        # Lower free energy = better state = BUY signal
        if agg_free_energy < -2.0:
            action = "BUY"
        elif agg_free_energy < -0.5:
            action = "BUY"
        elif agg_free_energy < 1.0:
            action = "HOLD"
        elif agg_free_energy < 3.0:
            action = "REDUCE"
        else:
            action = "SELL"
        
        # Calculate action probabilities (simplified)
        # Lower free energy = higher BUY probability
        prob_buy = max(0, min(1, (5 - agg_free_energy) / 10))
        prob_sell = max(0, min(1, (agg_free_energy + 3) / 10))
        prob_hold = 1 - prob_buy - prob_sell
        
        # Position sizing: positive = long, negative = short
        position = np.tanh(-agg_free_energy / 3)  # Range: -1 to 1
        
        return {
            "action": action,
            "action_index": 0 if action == "BUY" else (1 if action == "HOLD" else 2),
            "free_energy": agg_free_energy,
            "surprise": agg_surprise,
            "epistemic": agg_epistemic,
            "position": position,
            "action_probabilities": [max(0, prob_buy), max(0, prob_hold), max(0, prob_sell)],
            "window_signals": window_signals,
            "weights": weights,
            "pragmatic_value": -agg_surprise,
            "error": None
        }
        
    except Exception as e:
        return {
            "action": "HOLD",
            "free_energy": 0.0,
            "surprise": 0.0,
            "epistemic": 0.0,
            "position": 0.0,
            "action_probabilities": [0.33, 0.33, 0.34],
            "window_signals": [],
            "error": str(e)
        }


def compute_agent_signal(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    agent_config: Dict
) -> Dict:
    """Wrapper for FEP signal computation."""
    return compute_fep_signals(prices, macro_df, agent_config, train_agent=True)


def compute_cross_sectional_zscore(scores: Dict[str, float]) -> Dict[str, float]:
    """Compute cross-sectional z-scores within a universe."""
    values = np.array([v for v in scores.values() if not np.isnan(v)])
    if len(values) < 2:
        return {t: 0.0 for t in scores.keys()}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0 or np.isnan(std):
        return {t: 0.0 for t in scores.keys()}
    
    return {t: (scores[t] - mean) / std if not np.isnan(scores[t]) else 0.0 
            for t in scores.keys()}
