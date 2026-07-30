# P2-FEP-AGENT

**Free Energy-Principled Trading Agent**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine implements an **Active Inference** trading agent that explicitly minimizes **variational free energy** (surprise) using a generative model of market dynamics.

### Theory

The agent maintains a **generative model** of:
- Market returns (observation likelihood)
- Latent state transitions (market regimes)
- Trading impact (action effects)

It chooses actions that minimize **expected free energy**:
F = E[surprise] + KL[posterior || prior] - epistemic_value

text

**Two Drives:**
- **Pragmatic Drive**: Minimizes prediction error (exploitation)
- **Epistemic Drive**: Reduces model uncertainty (exploration)

**Key Metrics:**
- **Free Energy**: Lower = agent is more confident
- **Surprise**: Lower = predictions match reality
- **Epistemic**: Higher = more uncertainty (needs exploration)

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Windows

The agent maintains separate models for each time horizon:

| Window | Purpose |
|--------|---------|
| 63d | Short-term dynamics |
| 252d | Core signal (Primary - 40% weight) |
| 504d | Medium-term structure |
| 1008d | Structural regimes |
| 2016d | Secular trends |
| 4032d+ | Full history |

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| state_dim | Latent state dimension | 16 |
| ensemble_size | Number of models in ensemble | 5 |
| beta | Temperature for action selection | 1.0 |
| lambda_epistemic | Weight for exploration | 0.5 |
| lambda_pragmatic | Weight for exploitation | 0.5 |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-FEP-AGENT
cd P2-FEP-AGENT
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN

text

---

## Complete File Structure
P2-FEP-AGENT/
├── README.md ✅ Complete
├── config.py ✅ Complete (with Windows)
├── data_manager.py ✅ Complete
├── fep_agent.py ✅ Complete (with Multi-Window)
├── trainer.py ✅ Complete
├── push_results.py ✅ Complete
├── streamlit_app.py ✅ Complete (3 tabs)
├── us_calendar.py ✅ Complete
├── requirements.txt ✅ Complete
└── .github/
└── workflows/
└── daily.yml ✅ Complete

text

The FEP Agent now supports **all windows** (63, 252, 504, 1008, 2016, 4032, 4536) with the 252d window as the primary (40% weight). Each window has its own ensemble of generative models, and the agent aggregates their free energy estimates for unified decision-making!
