import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os
import numpy as np

st.set_page_config(page_title="Free Energy-Principled Agent", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #8e44ad}
.buy-card{background:linear-gradient(135deg,#1a472a 0%,#2d6a4f 60%,#40916c 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 6px 20px rgba(39,174,96,0.3)}
.sell-card{background:linear-gradient(135deg,#4a1a1a 0%,#6a2d2d 60%,#914040 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(231,76,60,0.3)}
.hold-card{background:linear-gradient(135deg,#2c3e50 0%,#4a5d6a 60%,#5d7a8a 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(44,62,80,0.3)}
.agent-card{background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 60%,#4a2c8a 100%);
            color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
            box-shadow:0 6px 20px rgba(142,68,173,0.4)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.metric-box{background:#f8f9fa;border-radius:10px;padding:0.8rem;margin:0.3rem 0;
            border-left:4px solid #8e44ad}
.metric-label{font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.metric-value{font-size:1.1rem;font-weight:700;color:#1a1a2e}
.window-badge{background:#8e44ad;border-radius:12px;padding:2px 10px;font-size:0.65rem;
              color:white;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Free Energy-Principled Agent</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Active Inference · Variational Free Energy Minimization · '
    'Pragmatic + Epistemic Drives · Unified Risk Management</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def action_badge(action: str) -> str:
    if "BUY" in action:
        return f'<span class="badge-buy">🟢 {action}</span>'
    elif "SELL" in action:
        return f'<span class="badge-sell">🔴 {action}</span>'
    else:
        return f'<span class="badge-hold">🟡 {action}</span>'

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


@st.cache_data(ttl=3600)
def list_repo_files():
    if not HF_TOKEN:
        st.sidebar.warning("⚠️ HF_TOKEN not set")
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        return api.list_repo_files(repo_id=RESULTS_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json_from_hf(path):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    try:
        api = HfApi(token=HF_TOKEN)
        content = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path, repo_type="dataset", token=HF_TOKEN)
        with open(content, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🧠 FEP Agent")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Primary Window:** {config.PRIMARY_WINDOW}d")
st.sidebar.markdown(f"**State Dim:** {config.GENERATIVE_MODEL['state_dim']}")
st.sidebar.markdown(f"**Ensemble Size:** {config.EPISTEMIC_DRIVE['ensemble_size']}")
st.sidebar.markdown(f"**β (Temperature):** {config.FREE_ENERGY['beta']}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Free Energy Weights:**")
st.sidebar.markdown(f"  • Pragmatic: {config.FREE_ENERGY['lambda_pragmatic']:.0%}")
st.sidebar.markdown(f"  • Epistemic: {config.FREE_ENERGY['lambda_epistemic']:.0%}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()
if not files:
    st.error("No files found. Run trainer.py first.")
    st.info(f"Looking in: {RESULTS_REPO}")
    st.stop()

tab1_path = find_latest(files, "fep_agent_")
tab2_path = find_latest(files, "fep_agent_breakdown_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json_from_hf(tab1_path)
if "error" in data1:
    st.error(f"Error: {data1['error']}")
    st.stop()

data2 = load_json_from_hf(tab2_path) if tab2_path else None
universes1 = data1.get("universes", {})
universes2 = data2.get("universes", {}) if data2 and "error" not in data2 else None

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.success(f"✅ {len(universes1)} universes")

tab1, tab2, tab3 = st.tabs(["🏆 Top Signals", "🔍 Full Breakdown", "📊 Window Analysis"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - TOP SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top Agent Signals — Buys & Sells")

    with st.expander("📖 How Free Energy-Principled Agent Works", expanded=True):
        st.markdown("""
**Free Energy-Principled Agent** uses Active Inference to make trading decisions:

| Component | What it does |
|-----------|--------------|
| **Generative Model** | Learns market dynamics and trading impact |
| **Pragmatic Drive** | Minimizes prediction error (exploitation) |
| **Epistemic Drive** | Reduces model uncertainty (exploration) |
| **Free Energy** | Unified objective combining both drives |

**Action Selection:**
- Agent chooses actions that minimize expected free energy
- BUY when future surprise is expected to decrease
- SELL when future surprise is expected to increase
- HOLD when uncertainty is too high

**Key Metrics:**
- **Free Energy**: Lower = better (agent is confident)
- **Surprise**: Lower = better (predictions match reality)
- **Epistemic**: Higher = more uncertainty (exploration needed)
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        top_buys = uni_data.get("top_buys", [])
        top_sells = uni_data.get("top_sells", [])
        
        if not top_buys and not top_sells:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Show Top Buys
        st.markdown(f'<div class="uni-title">🟢 {label} — Top Buys (Low Free Energy)</div>', unsafe_allow_html=True)
        if top_buys:
            cols = st.columns(3)
            for idx, item in enumerate(top_buys[:3]):
                ticker = item["ticker"]
                z_score = safe_float(item.get("z_score", 0))
                full_data = uni_data.get("full_scores", {}).get(ticker, {})
                action = full_data.get("action", "HOLD")
                free_energy = safe_float(full_data.get("free_energy", 0))
                surprise = safe_float(full_data.get("surprise", 0))
                epistemic = safe_float(full_data.get("epistemic", 0))
                position = safe_float(full_data.get("position", 0))
                
                with cols[idx]:
                    st.markdown(f"""
<div class="buy-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">F = {free_energy:.3f} | S = {surprise:.3f}</div>
  <div class="score">Epistemic = {epistemic:.3f}</div>
  <div class="score">Position = {position:.1%}</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No BUY signals in this universe")

        # Show Top Sells
        st.markdown(f'<div class="uni-title-sell">🔴 {label} — Top Sells (High Free Energy)</div>', unsafe_allow_html=True)
        if top_sells:
            cols = st.columns(3)
            for idx, item in enumerate(top_sells[:3]):
                ticker = item["ticker"]
                z_score = safe_float(item.get("z_score", 0))
                full_data = uni_data.get("full_scores", {}).get(ticker, {})
                action = full_data.get("action", "HOLD")
                free_energy = safe_float(full_data.get("free_energy", 0))
                surprise = safe_float(full_data.get("surprise", 0))
                epistemic = safe_float(full_data.get("epistemic", 0))
                position = safe_float(full_data.get("position", 0))
                
                with cols[idx]:
                    st.markdown(f"""
<div class="sell-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">F = {free_energy:.3f} | S = {surprise:.3f}</div>
  <div class="score">Epistemic = {epistemic:.3f}</div>
  <div class="score">Position = {position:.1%}</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No SELL signals in this universe")

        # Full ranking
        with st.expander(f"📋 Full ranking — {label}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "z-score": round(safe_float(info.get("z_score", 0)), 4),
                        "Free Energy": round(safe_float(info.get("free_energy", 0)), 4),
                        "Surprise": round(safe_float(info.get("surprise", 0)), 4),
                        "Epistemic": round(safe_float(info.get("epistemic", 0)), 4),
                        "Position": f"{safe_float(info.get('position', 0))*100:.0f}%",
                        "Action": info.get("action", "HOLD")
                    })
                df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · Free Energy = Pragmatic + Epistemic · Lower is better")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - FULL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Full Agent Breakdown")

    if not universes2:
        st.warning("Breakdown data not found.")
        st.stop()

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = universes2.get(universe_name, {})
        ranking = uni_data.get("full_ranking", [])

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not ranking:
            st.info(f"No data for {universe_name}")
            st.divider()
            continue

        rows = []
        for item in ranking:
            rows.append({
                "ETF": item.get("ticker", ""),
                "z-score": round(safe_float(item.get("z_score", 0)), 4),
                "Free Energy": round(safe_float(item.get("free_energy", 0)), 4),
                "Surprise": round(safe_float(item.get("surprise", 0)), 4),
                "Epistemic": round(safe_float(item.get("epistemic", 0)), 4),
                "Position": f"{safe_float(item.get('position', 0))*100:.0f}%",
                "Signal": round(safe_float(item.get("signal", 0)), 4),
                "Action": item.get("action", "HOLD")
            })

        df = pd.DataFrame(rows).sort_values("z-score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data2.get('run_date','?')} · Free Energy = variational free energy (lower = better)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - WINDOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Window-Level Agent Analysis")

    st.markdown("""
    ### How Windows Work
    
    The agent maintains separate generative models for each time horizon:
    
    | Window | Purpose | Weight |
    |--------|---------|--------|
    | **63d** | Short-term dynamics | Shared (60%) |
    | **252d** | Core signal (Primary) | **40%** |
    | **504d** | Medium-term structure | Shared (60%) |
    | **1008d** | Structural regimes | Shared (60%) |
    | **2016d** | Secular trends | Shared (60%) |
    | **4032d+** | Full history | Shared (60%) |
    
    The primary window (252d) gets 40% weight, all others share 60%.
    """)

    if not universes2:
        st.warning("Window-level data not found.")
        st.stop()

    # Get window signals from the data
    window_data = {}
    for universe_name in UNIVERSE_ORDER:
        uni_data = universes2.get(universe_name, {})
        ranking = uni_data.get("full_ranking", [])
        
        for item in ranking:
            ticker = item.get("ticker", "")
            if ticker not in window_data:
                window_data[ticker] = {}
            # Check if window signals are available
            if "window_signals" in item:
                for ws in item["window_signals"]:
                    w = ws.get("window", 0)
                    if w not in window_data[ticker]:
                        window_data[ticker][w] = {
                            "free_energy": ws.get("free_energy", 0),
                            "surprise": ws.get("surprise", 0),
                            "epistemic": ws.get("epistemic", 0),
                            "action": ws.get("action", "HOLD")
                        }

    if not window_data:
        st.info("No window-level data available. Run trainer with multi-window support.")
        st.stop()

    # Select ticker
    ticker_options = sorted(window_data.keys())
    selected_ticker = st.selectbox("Select ETF to analyze", ticker_options)

    if selected_ticker:
        data = window_data.get(selected_ticker, {})
        
        if data:
            # Create DataFrame
            rows = []
            for w, vals in data.items():
                rows.append({
                    "Window (days)": w,
                    "Free Energy": round(safe_float(vals.get("free_energy", 0)), 4),
                    "Surprise": round(safe_float(vals.get("surprise", 0)), 4),
                    "Epistemic": round(safe_float(vals.get("epistemic", 0)), 4),
                    "Action": vals.get("action", "HOLD")
                })
            
            df = pd.DataFrame(rows).sort_values("Window (days)")
            
            st.markdown(f"### 🧠 {selected_ticker} — Agent State by Window")
            
            # Color coding for free energy
            styled_df = df.style.map(
                lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x < 0 else '',
                subset=['Free Energy']
            ).map(
                lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and 0 <= x <= 1 else '',
                subset=['Free Energy']
            ).map(
                lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x > 1 else '',
                subset=['Free Energy']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Summary
            st.caption(f"**Summary:** Best window for {selected_ticker}: {df.loc[df['Free Energy'].idxmin(), 'Window (days)']}d (Free Energy = {df['Free Energy'].min():.4f})")
            
            # Add explanation
            with st.expander("📖 How to read window analysis"):
                st.markdown("""
                - **Free Energy**: Lower = better. Agent is more confident in this window.
                - **Surprise**: Lower = better. Predictions match reality better.
                - **Epistemic**: Higher = more uncertainty. Agent needs more exploration.
                - **Action**: What the agent would do based on this window alone.
                
                **Best Window** = lowest Free Energy → most reliable signal for this ETF.
                """)
        else:
            st.info(f"No window data available for {selected_ticker}")

    st.caption(f"Run date: {data2.get('run_date','?')} · Window-specific agent states")
