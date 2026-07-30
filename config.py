"""
config.py  —  Configuration for Free Energy-Principled Agent
=============================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - GENERATIVE_MODEL: model architecture parameters
  - FREE_ENERGY: variational free energy parameters
  - ACTION_SPACE: trading action constraints
  - EPISTEMIC_DRIVE: exploration parameters
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-fep-agent-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Generative Model ─────────────────────────────────────────────────────────

GENERATIVE_MODEL = {
    "state_dim": 16,           # Latent state dimension
    "observation_dim": 10,     # Market features (returns, volume, macro)
    "action_dim": 3,           # {BUY, HOLD, SELL} or continuous
    "hidden_dim": 64,          # Neural network hidden size
    "learning_rate": 0.001,    # Online learning rate
    "memory_size": 1000,       # Experience replay buffer
    "prediction_horizon": 5,   # Steps ahead to predict
}


# ── Free Energy Parameters ──────────────────────────────────────────────────

FREE_ENERGY = {
    "beta": 1.0,               # Temperature parameter for action selection
    "gamma": 0.99,             # Discount factor for future free energy
    "lambda_epistemic": 0.5,   # Weight for epistemic drive (exploration)
    "lambda_pragmatic": 0.5,   # Weight for pragmatic drive (exploitation)
    "surprise_threshold": 2.0, # Threshold for anomalous observations
}


# ── Action Space ─────────────────────────────────────────────────────────────

ACTION_SPACE = {
    "action_type": "discrete",  # discrete or continuous
    "n_actions": 3,            # BUY, HOLD, SELL
    "position_limits": [-1.0, 1.0],  # Min/max portfolio weight
    "max_trade_size": 0.1,      # Maximum fraction of portfolio to trade
    "transaction_cost": 0.001,  # 10bps per trade
}


# ── Epistemic Drive ──────────────────────────────────────────────────────────

EPISTEMIC_DRIVE = {
    "uncertainty_measure": "ensemble_variance",  # or "entropy"
    "ensemble_size": 5,          # Number of models in ensemble
    "exploration_bonus": 0.1,    # Bonus for high-uncertainty actions
    "decay_rate": 0.99,          # Exploration decay over time
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
