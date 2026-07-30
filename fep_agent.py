"""
fep_agent.py  —  Free Energy-Principled Trading Agent
=======================================================

Uses statistical moments + macro signals + momentum to compute:
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
    """Compute statistical features from returns for a given window."""
    if len(returns) < window:
        window = len(returns)
    
    recent = returns[-window:]
    
    # Basic statistics
    mean = np.mean(recent)
    std = np.std(recent)
    skew = pd.Series(recent).skew() if len(recent) > 2 else 0
    kurt = pd.Series(recent).kurtosis() if len(recent) > 3 else 0
    
    # Rolling statistics (short vs long)
    short_window = min(20, len(recent))
    long_window = min(60, len(recent))
    
    short_mean = np.mean(recent[-short_window:]) if len(recent) >= short_window else mean
    long_mean = np.mean(recent[-long_window:]) if len(recent) >= long_window else mean
    
    short_std = np.std(recent[-short_window:]) if len(recent) >= short_window else std
    long_std = np.std(recent[-long_window:]) if len(recent) >= long_window else std
    
    # Momentum: short vs long
    momentum = short_mean - long_mean
    
    # Volatility ratio
    vol_ratio = short_std / (long_std + 1e-6)
    
    # Recent performance (last 5 days vs last 20 days)
    recent_perf = np.mean(recent[-5:]) if len(recent) >= 5 else mean
    medium_perf = np.mean(recent[-20:]) if len(recent) >= 20 else mean
    
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "short_mean": short_mean,
        "long_mean": long_mean,
        "momentum": momentum,
        "vol_ratio": vol_ratio,
        "recent_perf": recent_perf,
        "medium_perf": medium_perf,
        "n": len(recent)
    }


def compute_surprise(returns: np.ndarray, window: int = 252) -> float:
    """Compute surprise as deviation from expected returns."""
    if len(returns) < 20:
        return 0.5
    
    features = compute_returns_features(returns, window)
    
    # Expected return based on long-term mean
    expected = features["long_mean"]
    
    # Actual recent return
    actual = features["short_mean"]
    
    # Surprise = absolute deviation from expectation (normalized)
    surprise = abs(actual - expected) / (features["std"] + 1e-6)
    
    # Scale to 0-2 range
    return min(2.0, surprise)


def compute_epistemic_value(returns: np.ndarray, window: int = 252) -> float:
    """Compute epistemic value (model uncertainty)."""
    if len(returns) < 20:
        return 0.5
    
    features = compute_returns_features(returns, window)
    
    # Uncertainty measures:
    # 1. Volatility relative to history
    vol_uncertainty = features["std"] / (np.mean(features["std"]) + 1e-6)
    vol_uncertainty = min(2.0, vol_uncertainty)
    
    # 2. Kurtosis (tail risk)
    tail_uncertainty = abs(features["kurt"]) / 5 if not np.isnan(features["kurt"]) else 0.5
    tail_uncertainty = min(1.0, tail_uncertainty)
    
    # 3. Volatility ratio (regime change)
    regime_uncertainty = abs(features["vol_ratio"] - 1) * 1.5
    regime_uncertainty = min(1.0, regime_uncertainty)
    
    epistemic = (vol_uncertainty + tail_uncertainty + regime_uncertainty) / 3
    return min(1.0, max(0.1, epistemic))


def compute_momentum_signal(returns: np.ndarray, window: int = 252) -> float:
    """Compute momentum signal (-1 to 1)."""
    if len(returns) < 20:
        return 0
    
    features = compute_returns_features(returns, window)
    
    # Momentum = short-term - long-term mean (normalized)
    momentum = features["momentum"] / (features["std"] + 1e-6)
    
    # Recent performance boost
    recent_boost = (features["recent_perf"] - features["medium_perf"]) / (features["std"] + 1e-6)
    
    # Combine
    signal = momentum * 0.7 + recent_boost * 0.3
    
    # Clamp to -1 to 1
    return max(-1.0, min(1.0, signal))


def compute_free_energy_with_macro(
    returns: np.ndarray, 
    macro_signal: float = 0,
    window: int = 252
) -> Dict:
    """
    Compute free energy incorporating macro signals.
    """
    features = compute_returns_features(returns, window)
    
    # Base components
    surprise = compute_surprise(returns, window)
    epistemic = compute_epistemic_value(returns, window)
    momentum = compute_momentum_signal(returns, window)
    
    # Pragmatic value: low surprise = good, high momentum = good
    pragmatic_value = -surprise + momentum * 0.5
    
    # Epistemic drive: explore when uncertain
    epistemic_value = epistemic * 0.3
    
    # Macro influence: risk-on macro = positive signal
    macro_effect = macro_signal * 0.2
    
    # Free energy (lower = better)
    free_energy = pragmatic_value + epistemic_value + macro_effect
    
    # Scale to a reasonable range
    free_energy = free_energy * 2
    
    return {
        "free_energy": free_energy,
        "surprise": surprise,
        "epistemic": epistemic,
        "momentum": momentum,
        "pragmatic_value": pragmatic_value,
        "epistemic_value": epistemic_value,
        "macro_effect": macro_effect,
        "mean": features["mean"],
        "std": features["std"],
        "skew": features["skew"],
        "kurt": features["kurt"]
    }


def compute_macro_signal(macro_df: pd.DataFrame, idx: int) -> float:
    """Extract macro signal from macro dataframe."""
    if macro_df is None or len(macro_df) == 0:
        return 0
    
    try:
        # Get latest macro values
        if idx < len(macro_df):
            latest_macro = macro_df.iloc[idx] if idx > 0 else macro_df.iloc[0]
        else:
            latest_macro = macro_df.iloc[-1]
        
        # Combine macro signals (VIX, DXY, spreads)
        # VIX: high = risk-off
        vix = latest_macro.get("VIX", 20)
        vix_signal = - (vix - 20) / 20  # Normalized: VIX > 20 = negative
        
        # DXY: high = risk-off
        dxy = latest_macro.get("DXY", 100)
        dxy_signal = - (dxy - 100) / 10
        
        # Combined macro signal (-1 to 1)
        macro_signal = (vix_signal + dxy_signal) / 2
        
        return max(-1.0, min(1.0, macro_signal))
    except Exception:
        return 0


def compute_fep_signals(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    train_agent: bool = True
) -> Dict:
    """Compute Free Energy-Principled signals for a single ticker."""
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < 50:
        return {
            "action": "HOLD",
            "free_energy": 0.0,
            "surprise": 0.0,
            "epistemic": 0.0,
            "momentum": 0.0,
            "position": 0.0,
            "action_probabilities": [0.33, 0.33, 0.34],
            "window_signals": [],
            "error": "Insufficient data"
        }
    
    try:
        windows = config.get("windows", [63, 252, 504, 1008, 2016, 4032])
        primary_window = config.get("primary_window", 252)
        
        # Compute macro signal
        macro_signal = compute_macro_signal(macro_df, -1)
        
        window_signals = []
        fe_values = []
        momentum_values = []
        
        for window in windows:
            fe_result = compute_free_energy_with_macro(
                returns, 
                macro_signal, 
                min(window, len(returns) - 1)
            )
            window_signals.append({
                "window": window,
                "free_energy": fe_result["free_energy"],
                "surprise": fe_result["surprise"],
                "epistemic": fe_result["epistemic"],
                "momentum": fe_result["momentum"],
                "mean": fe_result["mean"],
                "std": fe_result["std"]
            })
            fe_values.append(fe_result["free_energy"])
            momentum_values.append(fe_result["momentum"])
        
        # Weighted aggregate
        weights = {}
        for w in windows:
            if w == primary_window:
                weights[w] = 0.4
            else:
                weights[w] = 0.6 / (len(windows) - 1) if len(windows) > 1 else 0.6
        
        # Compute aggregate metrics
        agg_free_energy = 0
        agg_surprise = 0
        agg_epistemic = 0
        agg_momentum = 0
        
        for ws in window_signals:
            w = ws["window"]
            agg_free_energy += ws["free_energy"] * weights[w]
            agg_surprise += ws["surprise"] * weights[w]
            agg_epistemic += ws["epistemic"] * weights[w]
            agg_momentum += ws["momentum"] * weights[w]
        
        # Normalize free energy to a reasonable range
        agg_free_energy = max(-5, min(5, agg_free_energy))
        
        # Determine action based on free energy + momentum
        # Lower free energy = better state = BUY signal
        # Positive momentum = BUY signal
        combined_score = -agg_free_energy + agg_momentum * 0.5
        
        if combined_score > 1.5:
            action = "STRONG BUY"
        elif combined_score > 0.5:
            action = "BUY"
        elif combined_score > -0.5:
            action = "HOLD"
        elif combined_score > -1.5:
            action = "REDUCE"
        else:
            action = "STRONG SELL"
        
        # Calculate action probabilities (more differentiated)
        prob_buy = max(0, min(1, (combined_score + 2) / 4))
        prob_sell = max(0, min(1, (2 - combined_score) / 4))
        prob_hold = max(0, 1 - prob_buy - prob_sell)
        
        # Position sizing based on combined score
        position = np.tanh(combined_score / 2)  # Range: -1 to 1
        
        return {
            "action": action,
            "action_index": 0 if action in ["BUY", "STRONG BUY"] else (1 if action == "HOLD" else 2),
            "free_energy": agg_free_energy,
            "surprise": agg_surprise,
            "epistemic": agg_epistemic,
            "momentum": agg_momentum,
            "position": position,
            "action_probabilities": [float(prob_buy), float(prob_hold), float(prob_sell)],
            "window_signals": window_signals,
            "weights": weights,
            "macro_signal": macro_signal,
            "error": None
        }
        
    except Exception as e:
        return {
            "action": "HOLD",
            "free_energy": 0.0,
            "surprise": 0.0,
            "epistemic": 0.0,
            "momentum": 0.0,
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
