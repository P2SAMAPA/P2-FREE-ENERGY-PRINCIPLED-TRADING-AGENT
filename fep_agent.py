"""
fep_agent.py  —  Free Energy-Principled Agent (Proper Implementation)
========================================================================

Implements Active Inference with:
- Deep generative model (neural network)
- Variational inference (Bayesian updating)
- Expected free energy minimization
- Epistemic and pragmatic drives
- Proper action selection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ── Neural Network Models ─────────────────────────────────────────────────────

class GenerativeModel(nn.Module):
    """
    Deep generative model of market dynamics.
    
    p(o_t, s_t | s_{t-1}, a_{t-1}) = p(o_t | s_t) * p(s_t | s_{t-1}, a_{t-1})
    """
    
    def __init__(self, state_dim: int = 32, obs_dim: int = 16, action_dim: int = 3, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Transition model: p(s_t | s_{t-1}, a_{t-1})
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # mean + log_var
        )
        
        # Observation model: p(o_t | s_t)
        self.observation_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim * 2)  # mean + log_var
        )
        
        # Prior over initial state
        self.register_buffer('prior_mean', torch.zeros(state_dim))
        self.register_buffer('prior_log_var', torch.zeros(state_dim))
        
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute transition distribution p(s_t | s_{t-1}, a_{t-1})."""
        combined = torch.cat([state, action], dim=-1)
        params = self.transition_net(combined)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var
    
    def observation(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute observation distribution p(o_t | s_t)."""
        params = self.observation_net(state)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var
    
    def sample_state(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the generative model."""
        # Transition
        next_mean, next_log_var = self.transition(state, action)
        next_state = self.sample_state(next_mean, next_log_var)
        
        # Observation
        obs_mean, obs_log_var = self.observation(next_state)
        
        return next_mean, next_log_var, obs_mean, obs_log_var


class RecognitionModel(nn.Module):
    """
    Recognition model (inference network) for variational inference.
    q(s_t | o_t, s_{t-1}, a_{t-1})
    """
    
    def __init__(self, state_dim: int = 32, obs_dim: int = 16, action_dim: int = 3,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.recognition_net = nn.Sequential(
            nn.Linear(obs_dim + state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # mean + log_var
        )
    
    def forward(self, obs: torch.Tensor, prev_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution q(s_t | o_t, s_{t-1}, a_{t-1})."""
        combined = torch.cat([obs, prev_state, action], dim=-1)
        params = self.recognition_net(combined)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var


class FEPAgent:
    """
    Free Energy-Principled Agent with Deep Generative Model.
    
    Implements:
    - Variational free energy minimization
    - Expected free energy action selection
    - Epistemic (exploration) and pragmatic (exploitation) drives
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Dimensions
        self.state_dim = config.get("state_dim", 32)
        self.obs_dim = config.get("observation_dim", 16)
        self.action_dim = config.get("action_dim", 3)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.n_actions = config.get("n_actions", 3)
        self.action_labels = ["BUY", "HOLD", "SELL"]
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.beta = config.get("beta", 1.0)  # Temperature
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.lambda_epistemic = config.get("lambda_epistemic", 0.5)
        self.lambda_pragmatic = config.get("lambda_pragmatic", 0.5)
        
        # Initialize models
        self.generative_model = GenerativeModel(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.recognition_model = RecognitionModel(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generative_model.parameters(), lr=self.learning_rate)
        self.rec_optimizer = optim.Adam(self.recognition_model.parameters(), lr=self.learning_rate)
        
        # State tracking
        self.current_state = None
        self.current_obs = None
        self.current_action = None
        self.belief_mean = None
        self.belief_log_var = None
        self.position = 0.0
        self.max_position = 1.0
        
        # Memory buffer for online learning
        self.memory = []
        self.memory_size = config.get("memory_size", 1000)
        
        # Training counter
        self.training_steps = 0
        
    def reset_belief(self):
        """Reset the agent's belief state."""
        self.belief_mean = torch.zeros(self.state_dim)
        self.belief_log_var = torch.zeros(self.state_dim)
        self.current_state = torch.zeros(self.state_dim)
        
    def encode_observation(self, returns: np.ndarray, macro: np.ndarray) -> torch.Tensor:
        """Encode market data into observation tensor."""
        # Aggregate features
        features = []
        
        # Return features
        if len(returns) > 0:
            recent = returns[-min(20, len(returns)):]
            features.extend([
                np.mean(recent),
                np.std(recent),
                np.percentile(recent, 25),
                np.percentile(recent, 75),
                recent[-1] if len(recent) > 0 else 0,
                np.mean(recent[-5:]) if len(recent) >= 5 else 0,
                np.mean(recent[-10:]) if len(recent) >= 10 else 0,
            ])
        else:
            features.extend([0] * 7)
        
        # Macro features
        if len(macro) > 0:
            macro_flat = macro.flatten()[:9]
            features.extend(macro_flat.tolist())
        else:
            features.extend([0] * 9)
        
        # Pad or truncate to obs_dim
        if len(features) < self.obs_dim:
            features.extend([0] * (self.obs_dim - len(features)))
        else:
            features = features[:self.obs_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_variational_free_energy(self, obs: torch.Tensor, prev_state: torch.Tensor, 
                                         action: torch.Tensor) -> Dict:
        """
        Compute variational free energy for a given action.
        
        F = E_q[log q(s) - log p(o, s)] 
          = KL[q(s) || p(s)] - E_q[log p(o | s)]
        """
        # Get posterior from recognition model
        post_mean, post_log_var = self.recognition_model(obs, prev_state, action)
        post_std = torch.exp(0.5 * post_log_var)
        
        # Sample from posterior
        eps = torch.randn_like(post_std)
        state = post_mean + eps * post_std
        
        # Get prior from generative model
        prior_mean, prior_log_var = self.generative_model.transition(prev_state, action)
        prior_std = torch.exp(0.5 * prior_log_var)
        
        # KL divergence: KL[q(s) || p(s)]
        # Analytical KL for Gaussians
        kl_div = torch.sum(
            prior_log_var - post_log_var + 
            (post_std**2 + (post_mean - prior_mean)**2) / (prior_std**2 + 1e-8) - 1
        ) * 0.5
        
        # Expected log-likelihood: E_q[log p(o | s)]
        obs_mean, obs_log_var = self.generative_model.observation(state)
        obs_std = torch.exp(0.5 * obs_log_var)
        
        # Negative log-likelihood (simplified)
        nll = torch.sum(
            (obs - obs_mean)**2 / (2 * obs_std**2 + 1e-8) + 
            0.5 * torch.log(obs_std**2 + 1e-8)
        )
        
        # Variational free energy
        free_energy = kl_div + nll
        
        return {
            "free_energy": free_energy.item(),
            "kl_div": kl_div.item(),
            "nll": nll.item(),
            "post_mean": post_mean.detach().numpy(),
            "post_log_var": post_log_var.detach().numpy(),
            "state": state.detach().numpy()
        }
    
    def compute_expected_free_energy(self, obs: torch.Tensor, state: torch.Tensor, 
                                      action: torch.Tensor) -> Dict:
        """
        Compute expected free energy for action selection.
        
        G = E[F] = pragmatic_value + epistemic_value
        """
        # Generate imaginary future
        next_mean, next_log_var, obs_mean, obs_log_var = self.generative_model(state, action)
        
        # Epistemic value: expected reduction in uncertainty
        # Higher uncertainty = higher epistemic value (exploration)
        epistemic_value = torch.exp(0.5 * next_log_var).mean().item()
        epistemic_value = min(1.0, epistemic_value)
        
        # Pragmatic value: expected reward (negative surprise)
        # Lower surprise = higher pragmatic value (exploitation)
        surprise = torch.sum(
            (obs - obs_mean)**2 / (2 * torch.exp(obs_log_var) + 1e-8)
        ).item()
        pragmatic_value = -surprise
        
        # Expected free energy
        expected_free_energy = (
            self.lambda_pragmatic * pragmatic_value + 
            self.lambda_epistemic * epistemic_value
        )
        
        return {
            "expected_free_energy": expected_free_energy,
            "pragmatic_value": pragmatic_value,
            "epistemic_value": epistemic_value,
            "surprise": surprise
        }
    
    def select_action(self, obs: torch.Tensor, explore: bool = True) -> Dict:
        """Select action by minimizing expected free energy."""
        if self.belief_mean is None:
            self.reset_belief()
        
        action_results = []
        
        for action_idx in range(self.n_actions):
            action_tensor = torch.zeros(self.action_dim)
            action_tensor[action_idx] = 1.0
            
            # Compute expected free energy for this action
            result = self.compute_expected_free_energy(
                obs, 
                self.belief_mean,
                action_tensor
            )
            result["action"] = action_idx
            result["action_label"] = self.action_labels[action_idx]
            action_results.append(result)
        
        # Convert to array
        e_fe_values = np.array([r["expected_free_energy"] for r in action_results])
        
        # Softmax for action probabilities (lower expected free energy = better)
        if explore:
            # Exploration: softmax with temperature
            e_fe_shifted = e_fe_values - np.max(e_fe_values)
            probs = np.exp(self.beta * e_fe_shifted) / np.sum(np.exp(self.beta * e_fe_shifted) + 1e-8)
        else:
            # Exploitation: greedy
            probs = np.zeros(self.n_actions)
            probs[np.argmax(e_fe_values)] = 1.0
        
        # Sample action
        selected_idx = np.random.choice(self.n_actions, p=probs)
        
        # Position limits
        if self.position >= self.max_position * 0.9 and selected_idx == 0:  # BUY
            selected_idx = 1
        if self.position <= -self.max_position * 0.9 and selected_idx == 2:  # SELL
            selected_idx = 1
        
        # Update position
        action_delta = [0.1, 0.0, -0.1][selected_idx]
        self.position = np.clip(self.position + action_delta, -self.max_position, self.max_position)
        
        result = action_results[selected_idx]
        result["selected"] = True
        result["position"] = self.position
        result["action_probabilities"] = probs.tolist()
        result["action_index"] = selected_idx
        
        # Store action for learning
        self.current_action = selected_idx
        
        return result
    
    def update_belief(self, obs: torch.Tensor, action: torch.Tensor):
        """Update belief state using variational inference."""
        if self.belief_mean is None:
            self.reset_belief()
        
        # Compute posterior
        post_mean, post_log_var = self.recognition_model(
            obs, 
            self.belief_mean,
            action
        )
        
        # Update belief
        self.belief_mean = post_mean.detach()
        self.belief_log_var = post_log_var.detach()
        
        # Sample new state
        std = torch.exp(0.5 * self.belief_log_var)
        eps = torch.randn_like(std)
        self.current_state = self.belief_mean + eps * std
    
    def learn(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor,
              reward: float = 0.0):
        """
        Online learning step using variational free energy minimization.
        """
        # Prepare action tensor
        action_tensor = torch.zeros(self.action_dim)
        if isinstance(action, int):
            action_tensor[action] = 1.0
        else:
            action_tensor = action
        
        # Compute variational free energy
        fe_result = self.compute_variational_free_energy(
            obs,
            self.belief_mean if self.belief_mean is not None else torch.zeros(self.state_dim),
            action_tensor
        )
        
        # Compute loss = free_energy - reward
        loss = fe_result["free_energy"] - reward
        
        # Backpropagate
        self.gen_optimizer.zero_grad()
        self.rec_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.recognition_model.parameters(), 1.0)
        
        self.gen_optimizer.step()
        self.rec_optimizer.step()
        
        # Update belief
        self.update_belief(obs, action_tensor)
        
        # Store in memory
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append({
            "obs": obs.detach().numpy(),
            "action": action_tensor.detach().numpy(),
            "next_obs": next_obs.detach().numpy(),
            "reward": reward,
            "fe": fe_result["free_energy"]
        })
        
        self.training_steps += 1
        
        return fe_result
    
    def batch_learn(self, batch_size: int = 32):
        """Batch learning from memory."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        total_loss = 0
        for item in batch:
            obs = torch.tensor(item["obs"], dtype=torch.float32)
            action = torch.tensor(item["action"], dtype=torch.float32)
            next_obs = torch.tensor(item["next_obs"], dtype=torch.float32)
            reward = item["reward"]
            
            # Compute free energy
            fe_result = self.compute_variational_free_energy(obs, self.belief_mean, action)
            loss = fe_result["free_energy"] - reward
            
            total_loss += loss
        
        # Average loss
        avg_loss = total_loss / batch_size
        
        # Backpropagate
        self.gen_optimizer.zero_grad()
        self.rec_optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.recognition_model.parameters(), 1.0)
        self.gen_optimizer.step()
        self.rec_optimizer.step()


# ── Wrapper Functions ─────────────────────────────────────────────────────────

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
            "error": "Insufficient data (need at least 100 days)"
        }
    
    try:
        # Initialize agent
        agent = FEPAgent(config)
        
        # Training phase
        if train_agent:
            # Use last 200 days for training
            train_len = min(200, len(returns) - 20)
            train_returns = returns[-train_len-20:]
            
            for i in range(10, len(train_returns) - 10, 2):
                # Prepare observations
                obs_window = train_returns[max(0, i-10):i+1]
                macro_window = macro[max(0, i-5):i+1] if len(macro) > 0 else np.zeros((1, 6))
                
                obs = agent.encode_observation(obs_window, macro_window)
                next_obs = agent.encode_observation(
                    train_returns[max(0, i-9):i+2],
                    macro_window
                )
                
                # Random action for exploration during training
                action = np.random.randint(0, agent.n_actions)
                
                # Learn
                agent.learn(obs, action, next_obs, reward=0.01)
                
                # Batch learn every 10 steps
                if i % 10 == 0 and len(agent.memory) > 32:
                    agent.batch_learn(batch_size=32)
        
        # Inference
        latest_returns = returns[-20:]
        latest_macro = macro[-5:] if len(macro) > 0 else np.zeros((1, 6))
        
        obs = agent.encode_observation(latest_returns, latest_macro)
        
        # Select action
        result = agent.select_action(obs, explore=False)
        
        # Compute surprise
        surprise = result.get("surprise", 0.0)
        
        # Get free energy from the agent's belief
        if agent.belief_mean is not None:
            free_energy = agent.compute_variational_free_energy(
                obs,
                agent.belief_mean,
                torch.zeros(agent.action_dim)
            )["free_energy"]
        else:
            free_energy = 0.0
        
        # Window-specific signals (for breakdown)
        window_signals = []
        for window in config.get("windows", [63, 252, 504, 1008]):
            # Quick approximation for each window
            window_returns = returns[-min(window, len(returns)):]
            window_macro = macro[-min(5, len(macro)):] if len(macro) > 0 else np.zeros((1, 6))
            window_obs = agent.encode_observation(window_returns, window_macro)
            
            # Compute rough free energy for this window
            if agent.belief_mean is not None:
                fe = agent.compute_variational_free_energy(
                    window_obs,
                    agent.belief_mean,
                    torch.zeros(agent.action_dim)
                )["free_energy"]
            else:
                fe = 0.0
            
            window_signals.append({
                "window": window,
                "free_energy": fe,
                "surprise": surprise,
                "epistemic": result.get("epistemic_value", 0.0),
                "action": result.get("action_label", "HOLD")
            })
        
        return {
            "action": result.get("action_label", "HOLD"),
            "action_index": result.get("action_index", 1),
            "free_energy": free_energy,
            "surprise": surprise,
            "epistemic": result.get("epistemic_value", 0.0),
            "position": agent.position,
            "action_probabilities": result.get("action_probabilities", [0.33, 0.33, 0.34]),
            "window_signals": window_signals,
            "pragmatic_value": result.get("pragmatic_value", 0.0),
            "epistemic_value": result.get("epistemic_value", 0.0),
            "error": None
        }
        
    except Exception as e:
        import traceback
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
