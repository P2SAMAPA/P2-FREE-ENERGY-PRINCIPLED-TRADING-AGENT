"""
fep_agent.py  —  Free Energy-Principled Trading Agent
=======================================================

Implements Active Inference for trading with multi-window support:
- Generative model trained on multiple time horizons
- Ensemble of models per window
- Unified free energy minimization across horizons
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class GenerativeModel:
    """
    Generative model of market dynamics for a specific window.
    """
    
    def __init__(self, state_dim: int = 16, obs_dim: int = 10, action_dim: int = 3):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Transition model: p(s_{t+1} | s_t, a_t)
        self.transition_mean = np.zeros((state_dim, state_dim + action_dim))
        self.transition_cov = np.eye(state_dim)
        
        # Observation model: p(o_t | s_t)
        self.obs_mean = np.zeros((obs_dim, state_dim))
        self.obs_cov = np.eye(obs_dim)
        
        # Prior over initial state
        self.prior_mean = np.zeros(state_dim)
        self.prior_cov = np.eye(state_dim)
        
        # Action effects
        self.action_effects = np.zeros((state_dim, action_dim))
        
        # Learning parameters
        self.learning_rate = 0.001
        self.steps = 0
        self.window = 252  # Default
        
    def set_window(self, window: int):
        """Set the training window for this model."""
        self.window = window
        # Adjust learning rate based on window size
        self.learning_rate = 0.001 * (252 / window)
        
    def predict_state(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state given current state and action."""
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        combined = np.concatenate([state, action_onehot])
        
        mean = self.transition_mean @ combined
        cov = self.transition_cov
        
        return mean, cov
    
    def predict_observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict observation given state."""
        mean = self.obs_mean @ state
        cov = self.obs_cov
        return mean, cov
    
    def update(self, state: np.ndarray, action: int, next_state: np.ndarray, 
               observation: np.ndarray, learning_rate: float = None):
        """Update model parameters using new data."""
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        combined = np.concatenate([state, action_onehot])
        
        # Update transition model (online learning)
        prediction = self.transition_mean @ combined
        error = next_state - prediction
        self.transition_mean += learning_rate * np.outer(error, combined)
        
        # Update observation model
        obs_pred = self.obs_mean @ state
        obs_error = observation - obs_pred
        self.obs_mean += learning_rate * np.outer(obs_error, state)
        
        self.steps += 1


class EnsembleModel:
    """
    Ensemble of generative models for uncertainty quantification.
    """
    
    def __init__(self, n_models: int = 5, state_dim: int = 16, 
                 obs_dim: int = 10, action_dim: int = 3, window: int = 252):
        self.models = [
            GenerativeModel(state_dim, obs_dim, action_dim)
            for _ in range(n_models)
        ]
        self.n_models = n_models
        self.window = window
        
        # Set window for all models
        for model in self.models:
            model.set_window(window)
        
    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions and uncertainty."""
        predictions = []
        for model in self.models:
            mean, _ = model.predict_state(state, action)
            predictions.append(mean)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        return mean, variance
    
    def predict_observation(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble observation predictions and uncertainty."""
        predictions = []
        for model in self.models:
            mean, _ = model.predict_observation(state)
            predictions.append(mean)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        return mean, variance
    
    def update(self, state: np.ndarray, action: int, next_state: np.ndarray,
               observation: np.ndarray, learning_rate: float = 0.001):
        """Update all models in ensemble."""
        for model in self.models:
            model.update(state, action, next_state, observation, 
                        learning_rate * np.random.uniform(0.8, 1.2))


class MultiWindowFEPAgent:
    """
    Free Energy-Principled Trading Agent with Multi-Window Support.
    
    Maintains separate ensembles for each window and combines their
    free energy estimates.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Windows
        self.windows = config.get("windows", [63, 252, 504, 1008, 2016, 4032])
        self.primary_window = config.get("primary_window", 252)
        
        # State dimensions
        self.state_dim = config.get("state_dim", 16)
        self.obs_dim = config.get("observation_dim", 10)
        self.action_dim = config.get("action_dim", 3)
        self.n_actions = config.get("n_actions", 3)
        self.action_labels = ["BUY", "HOLD", "SELL"]
        
        # Create ensembles for each window
        self.ensembles = {}
        for window in self.windows:
            self.ensembles[window] = EnsembleModel(
                n_models=config.get("ensemble_size", 5),
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                window=window
            )
        
        # Free energy parameters
        self.beta = config.get("beta", 1.0)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_epistemic = config.get("lambda_epistemic", 0.5)
        self.lambda_pragmatic = config.get("lambda_pragmatic", 0.5)
        self.surprise_threshold = config.get("surprise_threshold", 2.0)
        
        # Position tracking
        self.position = 0.0
        self.max_position = 1.0
        
        # History
        self.history = []
        self.action_history = []
        
    def encode_state(self, returns: np.ndarray, macro: np.ndarray, window: int) -> np.ndarray:
        """Encode market observations into latent state for a specific window."""
        # Take recent data based on window
        recent_returns = returns[-min(window, len(returns)):]
        
        if len(recent_returns) < 10:
            recent_returns = np.pad(recent_returns, (0, 10 - len(recent_returns)))
        
        # Aggregate features
        if len(recent_returns) > 20:
            # Use statistics for longer windows
            features = np.array([
                np.mean(recent_returns[-20:]),
                np.std(recent_returns[-20:]),
                np.mean(recent_returns[-60:]) if len(recent_returns) >= 60 else 0,
                np.std(recent_returns[-60:]) if len(recent_returns) >= 60 else 0,
                np.mean(recent_returns[-252:]) if len(recent_returns) >= 252 else 0,
                np.std(recent_returns[-252:]) if len(recent_returns) >= 252 else 0,
                recent_returns[-1] if len(recent_returns) > 0 else 0,
                recent_returns[-5] if len(recent_returns) >= 5 else 0,
                recent_returns[-10] if len(recent_returns) >= 10 else 0,
            ])
        else:
            # Use raw returns for short windows
            features = recent_returns[-10:]
            if len(features) < 10:
                features = np.pad(features, (0, 10 - len(features)))
        
        # Add macro features
        if len(macro) > 0:
            macro_features = macro[-6:] if len(macro) >= 6 else np.pad(macro, (0, 6 - len(macro)))
        else:
            macro_features = np.zeros(6)
        
        state = np.concatenate([features[:10], macro_features[:6]])
        
        # Ensure fixed dimension
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
            
        return state
    
    def compute_free_energy_for_window(self, state: np.ndarray, action: int, 
                                        window: int) -> Dict:
        """
        Compute variational free energy for a specific window.
        """
        ensemble = self.ensembles[window]
        
        # Predict next state
        next_mean, next_var = ensemble.predict(state, action)
        
        # Pragmatic value: expected reduction in surprise
        surprise = 0.5 * np.sum((next_mean - state) ** 2 / (next_var + 1e-6))
        
        # Epistemic value: reduction in model uncertainty
        epistemic = np.mean(next_var) / (np.mean(next_var) + 1e-6)
        
        # KL divergence (simplified)
        kl_div = 0.5 * np.sum((next_mean - ensemble.models[0].prior_mean) ** 2 / 
                               (ensemble.models[0].prior_cov.diagonal() + 1e-6))
        
        # Free energy components
        pragmatic_value = -surprise
        epistemic_value = epistemic * self.lambda_epistemic
        
        # Total expected free energy
        free_energy = (self.lambda_pragmatic * pragmatic_value + 
                       self.lambda_epistemic * epistemic_value -
                       self.beta * kl_div)
        
        return {
            "free_energy": free_energy,
            "surprise": surprise,
            "epistemic": epistemic,
            "kl_div": kl_div,
            "pragmatic_value": pragmatic_value,
            "epistemic_value": epistemic_value,
            "window": window,
            "action": action,
            "action_label": self.action_labels[action]
        }
    
    def compute_aggregate_free_energy(self, state: np.ndarray, action: int) -> Dict:
        """
        Compute aggregate free energy across all windows.
        """
        results = []
        for window in self.windows:
            result = self.compute_free_energy_for_window(state, action, window)
            results.append(result)
        
        # Weight windows (primary window gets higher weight)
        weights = {}
        for w in self.windows:
            if w == self.primary_window:
                weights[w] = 0.4
            else:
                # Other windows get equal weight summing to 0.6
                weights[w] = 0.6 / (len(self.windows) - 1) if len(self.windows) > 1 else 0.6
        
        # Aggregate
        agg_free_energy = sum(r["free_energy"] * weights[r["window"]] for r in results)
        agg_surprise = sum(r["surprise"] * weights[r["window"]] for r in results)
        agg_epistemic = sum(r["epistemic"] * weights[r["window"]] for r in results)
        
        return {
            "free_energy": agg_free_energy,
            "surprise": agg_surprise,
            "epistemic": agg_epistemic,
            "window_results": results,
            "weights": weights,
            "action": action,
            "action_label": self.action_labels[action]
        }
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Dict:
        """
        Select action that minimizes expected free energy.
        """
        action_values = []
        
        for action in range(self.n_actions):
            fe_result = self.compute_aggregate_free_energy(state, action)
            action_values.append(fe_result)
        
        # Compute action probabilities using softmax
        fe_values = np.array([a["free_energy"] for a in action_values])
        
        if explore:
            probs = softmax(self.beta * fe_values)
        else:
            probs = np.zeros(self.n_actions)
            probs[np.argmax(fe_values)] = 1.0
        
        # Sample action
        selected_action = np.random.choice(self.n_actions, p=probs)
        
        # Apply position limits
        if self.position >= self.max_position * 0.9 and selected_action == 0:  # BUY
            selected_action = 1  # HOLD
        if self.position <= -self.max_position * 0.9 and selected_action == 2:  # SELL
            selected_action = 1  # HOLD
        
        # Update position
        action_delta = [0.1, 0.0, -0.1][selected_action]
        self.position = np.clip(self.position + action_delta, 
                               -self.max_position, self.max_position)
        
        result = action_values[selected_action]
        result["selected"] = True
        result["position"] = self.position
        result["action_probabilities"] = probs.tolist()
        
        # Record history
        self.action_history.append({
            "action": selected_action,
            "position": self.position,
            "free_energy": result["free_energy"],
            "surprise": result["surprise"],
            "epistemic": result["epistemic"]
        })
        
        return result
    
    def update_model(self, state: np.ndarray, action: int, 
                     next_state: np.ndarray, observation: np.ndarray):
        """Update generative models with new experience."""
        for window, ensemble in self.ensembles.items():
            # Use window-specific learning rate
            lr = 0.001 * (252 / window)
            ensemble.update(state, action, next_state, observation, lr)
    
    def compute_surprise(self, observation: np.ndarray) -> float:
        """Compute surprise of an observation (aggregate across windows)."""
        if self.position == 0:
            state = np.zeros(self.state_dim)
        else:
            state = np.ones(self.state_dim) * self.position
        
        surprises = []
        for window, ensemble in self.ensembles.items():
            mean, cov = ensemble.predict_observation(state)
            diff = observation - mean
            inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(len(mean)))
            surprise = diff @ inv_cov @ diff
            surprises.append(surprise)
        
        return float(np.mean(surprises))


def compute_fep_signals(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    train_agent: bool = True
) -> Dict:
    """
    Compute Free Energy-Principled signals for a single ticker.
    """
    # Prepare data
    returns = np.log(prices / prices.shift(1)).dropna().values
    macro = macro_df.values
    
    if len(returns) < 100:
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
    
    # Initialize agent
    agent = MultiWindowFEPAgent(config)
    
    # Training phase (online learning)
    if train_agent:
        # Use a rolling window for training
        train_window = min(252, len(returns) - 1)
        train_returns = returns[-train_window-20:]
        
        for i in range(len(train_returns) - 20):
            # Encode state
            window_returns = train_returns[max(0, i-10):i+1]
            window_macro = macro[max(0, i-5):i+1] if len(macro) > 0 else np.zeros((1, 6))
            
            # Use primary window for training state
            state = agent.encode_state(window_returns, window_macro.flatten()[:6] if len(window_macro) > 0 else np.zeros(6), 
                                      agent.primary_window)
            
            # Random action for exploration during training
            action = np.random.randint(0, agent.n_actions)
            
            # Next state
            next_window_returns = train_returns[max(0, i-9):i+2]
            next_state = agent.encode_state(next_window_returns, window_macro.flatten()[:6] if len(window_macro) > 0 else np.zeros(6),
                                           agent.primary_window)
            
            # Observation
            obs = np.concatenate([train_returns[i:i+1], window_macro.flatten()[:9] if len(window_macro) > 0 else np.zeros(9)])
            
            # Update model
            agent.update_model(state, action, next_state, obs[:agent.obs_dim])
    
    # Latest state for inference
    latest_returns = returns[-20:]
    latest_macro = macro[-5:] if len(macro) > 0 else np.zeros((1, 6))
    
    state = agent.encode_state(latest_returns, latest_macro.flatten()[:6] if len(latest_macro) > 0 else np.zeros(6),
                              agent.primary_window)
    
    # Select action
    result = agent.select_action(state, explore=False)
    
    # Compute surprise
    obs = np.concatenate([returns[-1:], latest_macro.flatten()[:9] if len(latest_macro) > 0 else np.zeros(9)])
    surprise = agent.compute_surprise(obs[:agent.obs_dim])
    
    # Get window-specific signals
    window_signals = []
    if "window_results" in result:
        for wr in result["window_results"]:
            window_signals.append({
                "window": wr["window"],
                "free_energy": wr["free_energy"],
                "surprise": wr["surprise"],
                "epistemic": wr["epistemic"],
                "action": wr["action_label"]
            })
    
    return {
        "action": result["action_label"],
        "action_index": result["action"],
        "free_energy": result["free_energy"],
        "surprise": surprise,
        "epistemic": result["epistemic"],
        "position": result["position"],
        "action_probabilities": result["action_probabilities"],
        "window_signals": window_signals,
        "weights": result.get("weights", {}),
        "pragmatic_value": result.get("pragmatic_value", 0),
        "epistemic_value": result.get("epistemic_value", 0),
        "kl_div": result.get("kl_div", 0)
    }


def compute_agent_signal(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    agent_config: Dict
) -> Dict:
    """
    Wrapper for FEP signal computation.
    """
    try:
        result = compute_fep_signals(prices, macro_df, agent_config)
        return result
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


def compute_cross_sectional_zscore(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Compute cross-sectional z-scores within a universe.
    """
    values = np.array([v for v in scores.values() if not np.isnan(v)])
    if len(values) < 2:
        return {t: 0.0 for t in scores.keys()}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0 or np.isnan(std):
        return {t: 0.0 for t in scores.keys()}
    
    return {t: (scores[t] - mean) / std if not np.isnan(scores[t]) else 0.0 
            for t in scores.keys()}
