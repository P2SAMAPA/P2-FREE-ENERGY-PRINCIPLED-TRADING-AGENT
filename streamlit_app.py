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
.top-card{background:linear-gradient(135deg,#0d47a1 0%,#1565c0 60%,#1e88e5 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 6px 20px rgba(21,101,192,0.4)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-top{background:#1a237e;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
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

def color_by_score(val, reverse=False):
    """Color code from green (best) to red (worst)."""
    if isinstance(val, (int, float)):
        if val > 0.5:
            return 'background-color: #27ae60; color: white;'
        elif val > 0.2:
            return 'background-color: #2ecc71; color: white;'
        elif val > -0.2:
            return 'background-color: #f1c40f; color: black;'
        elif val > -0.5:
            return 'background-color: #e67e22; color: white;'
        else:
            return 'background-color: #e74c3c; color: white;'
    return ''


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
st.sidebar.markdown(f"  • Momentum: {config.FREE_ENERGY['lambda_momentum']:.0%}")
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏆 Top 3 Buys & Full Ranking", "🔍 Window-by-Window Analysis"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - TOP 3 BUYS & FULL RANKING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top 3 ETFs to BUY — Ranked by Composite Score")

    with st.expander("📖 How the Agent Works", expanded=False):
        st.markdown("""
**Free Energy-Principled Agent** uses Active Inference to make trading decisions:

| Component | What it does |
|-----------|--------------|
| **Generative Model** | Learns market dynamics and trading impact |
| **Pragmatic Drive** | Minimizes prediction error (exploitation) |
| **Epistemic Drive** | Reduces model uncertainty (exploration) |
| **Free Energy** | Unified objective combining both drives |

**Composite Score** = z-score (40%) + Free Energy (30%) + Epistemic (20%) + Surprise (10%)
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        full_scores = uni_data.get("full_scores", {})
        
        if not full_scores:
            st.warning(f"No data available for {UNIVERSE_LABELS.get(universe_name, universe_name)}")
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Compute composite scores
        ranked_etfs = []
        for ticker, info in full_scores.items():
            z_score = safe_float(info.get("z_score", 0))
            free_energy = safe_float(info.get("free_energy", 0))
            surprise = safe_float(info.get("surprise", 0))
            epistemic = safe_float(info.get("epistemic", 0))
            action = info.get("action", "HOLD")
            
            # Composite: z-score (40%) + free_energy_reversed (30%) + epistemic (20%) + surprise_reversed (10%)
            free_energy_norm = -np.clip(free_energy / 2, -1, 1)
            surprise_norm = -np.clip(surprise / 2, -1, 1)
            epistemic_norm = np.clip(epistemic, 0, 1)
            
            composite = (0.40 * z_score + 
                        0.30 * free_energy_norm + 
                        0.20 * epistemic_norm + 
                        0.10 * surprise_norm)
            composite = np.clip(composite, -1, 1)
            
            ranked_etfs.append({
                "ticker": ticker,
                "z_score": z_score,
                "free_energy": free_energy,
                "surprise": surprise,
                "epistemic": epistemic,
                "composite": composite,
                "action": action
            })
        
        # Sort by composite (highest = best to buy)
        ranked_etfs = sorted(ranked_etfs, key=lambda x: x["composite"], reverse=True)
        top_3 = ranked_etfs[:3]
        
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)
        
        # ── TOP 3 BUYS ──────────────────────────────────────────────────────────
        st.markdown("#### 🟢 Top 3 ETFs to BUY")
        cols = st.columns(3)
        for idx, etf in enumerate(top_3):
            ticker = etf["ticker"]
            composite = etf["composite"]
            z_score = etf["z_score"]
            free_energy = etf["free_energy"]
            action = etf["action"]
            
            rank_badge = "⭐" if idx == 0 else ("🥈" if idx == 1 else "🥉")
            
            with cols[idx]:
                st.markdown(f"""
<div class="top-card">
  <div class="ticker">{rank_badge} {ticker}</div>
  <div class="score">Composite = {composite:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">Free Energy = {free_energy:.3f}</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        
        # ── FULL RANKING TABLE ──────────────────────────────────────────────────
        with st.expander(f"📋 Full Ranking — {label} (Green = Best to Buy, Red = Worst)"):
            rows = []
            for idx, etf in enumerate(ranked_etfs):
                rows.append({
                    "Rank": idx + 1,
                    "ETF": etf["ticker"],
                    "Composite Score": round(etf["composite"], 4),
                    "z-score": round(etf["z_score"], 4),
                    "Free Energy": round(etf["free_energy"], 4),
                    "Surprise": round(etf["surprise"], 4),
                    "Epistemic": round(etf["epistemic"], 4),
                    "Action": etf["action"]
                })
            
            df = pd.DataFrame(rows)
            
            # Apply color coding
            styled_df = df.style.map(
                lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x <= 3 else '',
                subset=['Rank']
            ).map(
                lambda x: 'background-color: #2ecc71; color: white;' if isinstance(x, (int, float)) and 4 <= x <= 6 else '',
                subset=['Rank']
            ).map(
                lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and 7 <= x <= 10 else '',
                subset=['Rank']
            ).map(
                lambda x: 'background-color: #e67e22; color: white;' if isinstance(x, (int, float)) and 11 <= x <= 15 else '',
                subset=['Rank']
            ).map(
                lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x > 15 else '',
                subset=['Rank']
            )
            
            # Color composite column
            styled_df = styled_df.map(
                lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.3 else '',
                subset=['Composite Score']
            ).map(
                lambda x: 'background-color: #2ecc71; color: white;' if isinstance(x, (int, float)) and 0.1 < x <= 0.3 else '',
                subset=['Composite Score']
            ).map(
                lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.1 < x <= 0.1 else '',
                subset=['Composite Score']
            ).map(
                lambda x: 'background-color: #e67e22; color: white;' if isinstance(x, (int, float)) and -0.3 < x <= -0.1 else '',
                subset=['Composite Score']
            ).map(
                lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.3 else '',
                subset=['Composite Score']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Summary stats
            if ranked_etfs:
                best = ranked_etfs[0]
                worst = ranked_etfs[-1]
                st.caption(f"**Best:** {best['ticker']} (Composite: {best['composite']:+.3f}) | **Worst:** {worst['ticker']} (Composite: {worst['composite']:+.3f})")
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · Composite = z-score(40%) + Free Energy(30%) + Epistemic(20%) + Surprise(10%)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - WINDOW-BY-WINDOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Window-by-Window Analysis — All ETFs Color-Coded")

    if not universes2:
        st.warning("Window-level data not found.")
        st.stop()

    # Get all window signals from the data
    all_window_data = {}
    window_columns = {}
    
    for universe_name in UNIVERSE_ORDER:
        uni_data = universes2.get(universe_name, {})
        ranking = uni_data.get("full_ranking", [])
        
        for item in ranking:
            ticker = item.get("ticker", "")
            window_signals = item.get("window_signals", [])
            
            if ticker not in all_window_data:
                all_window_data[ticker] = {
                    "universe": universe_name,
                    "z_score": safe_float(item.get("z_score", 0)),
                    "action": item.get("action", "HOLD")
                }
            
            for ws in window_signals:
                w = ws.get("window", 0)
                if w not in window_columns:
                    window_columns[w] = {
                        "free_energy": f"FE_{w}",
                        "surprise": f"S_{w}",
                        "epistemic": f"E_{w}",
                        "action": f"A_{w}"
                    }
                
                all_window_data[ticker][f"FE_{w}"] = safe_float(ws.get("free_energy", 0))
                all_window_data[ticker][f"S_{w}"] = safe_float(ws.get("surprise", 0))
                all_window_data[ticker][f"E_{w}"] = safe_float(ws.get("epistemic", 0))
                all_window_data[ticker][f"A_{w}"] = ws.get("action", "HOLD")

    if not all_window_data:
        st.warning("No window-level data available. Run trainer.py with multi-window support.")
        st.info("Check that your data has window_signals in the JSON output.")
        st.stop()

    # Convert to DataFrame
    df_rows = []
    for ticker, data in all_window_data.items():
        row = {
            "ETF": ticker,
            "Universe": data.get("universe", ""),
            "z-score": round(data.get("z_score", 0), 4),
            "Action": data.get("action", "HOLD")
        }
        for w in sorted(window_columns.keys()):
            row[f"FE_{w}"] = round(data.get(f"FE_{w}", 0), 4)
            row[f"S_{w}"] = round(data.get(f"S_{w}", 0), 4)
            row[f"E_{w}"] = round(data.get(f"E_{w}", 0), 4)
            row[f"A_{w}"] = data.get(f"A_{w}", "HOLD")
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    
    if df.empty:
        st.warning("No data available to display.")
        st.stop()
    
    df = df.sort_values("z-score", ascending=False)

    # ── Select Universe Filter ──────────────────────────────────────────────────
    universe_filter = st.selectbox(
        "Filter by Universe",
        ["All Universes"] + [UNIVERSE_LABELS.get(u, u) for u in UNIVERSE_ORDER if u in df["Universe"].values]
    )

    if universe_filter != "All Universes":
        for key, label in UNIVERSE_LABELS.items():
            if label == universe_filter:
                df = df[df["Universe"] == key]
                break

    # ── Window Selector ─────────────────────────────────────────────────────────
    sorted_windows = sorted(window_columns.keys())
    display_windows = []
    for w in sorted_windows:
        if w == config.PRIMARY_WINDOW:
            display_windows.append(f"{w}d ⭐ Primary")
        else:
            display_windows.append(f"{w}d")
    
    selected_window_idx = st.selectbox(
        "Select Window to Display",
        options=range(len(display_windows)),
        format_func=lambda i: display_windows[i]
    )
    selected_window = sorted_windows[selected_window_idx]

    # ── Display Window Data ────────────────────────────────────────────────────
    st.markdown(f"### 📊 Window: **{selected_window}d** — All ETFs Color-Coded")

    # Prepare columns for display
    display_rows = []
    for idx, row in df.iterrows():
        display_rows.append({
            "Rank": len(display_rows) + 1,
            "ETF": row["ETF"],
            "Universe": row["Universe"],
            "z-score": row["z-score"],
            "Action": row["Action"],
            f"Free Energy ({selected_window}d)": row[f"FE_{selected_window}"],
            f"Surprise ({selected_window}d)": row[f"S_{selected_window}"],
            f"Epistemic ({selected_window}d)": row[f"E_{selected_window}"],
            f"Action @ {selected_window}d": row[f"A_{selected_window}"]
        })

    if not display_rows:
        st.warning(f"No data available for window {selected_window}d")
        st.stop()
    
    display_df = pd.DataFrame(display_rows).sort_values("z-score", ascending=False)
    display_df["Rank"] = range(1, len(display_df) + 1)

    # Color code by z-score (green to red)
    styled_df = display_df.style.map(
        lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x <= 3 else '',
        subset=['Rank']
    ).map(
        lambda x: 'background-color: #2ecc71; color: white;' if isinstance(x, (int, float)) and 4 <= x <= 6 else '',
        subset=['Rank']
    ).map(
        lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and 7 <= x <= 10 else '',
        subset=['Rank']
    ).map(
        lambda x: 'background-color: #e67e22; color: white;' if isinstance(x, (int, float)) and 11 <= x <= 15 else '',
        subset=['Rank']
    ).map(
        lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x > 15 else '',
        subset=['Rank']
    )

    # Color z-score column
    styled_df = styled_df.map(
        lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.5 else '',
        subset=['z-score']
    ).map(
        lambda x: 'background-color: #2ecc71; color: white;' if isinstance(x, (int, float)) and 0.2 < x <= 0.5 else '',
        subset=['z-score']
    ).map(
        lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.2 < x <= 0.2 else '',
        subset=['z-score']
    ).map(
        lambda x: 'background-color: #e67e22; color: white;' if isinstance(x, (int, float)) and -0.5 < x <= -0.2 else '',
        subset=['z-score']
    ).map(
        lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.5 else '',
        subset=['z-score']
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Top 3 at This Window ────────────────────────────────────────────────────
    st.markdown(f"### 🟢 Top 3 ETFs to BUY at {selected_window}d")

    top_3_window = display_df.head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_3_window.iterrows()):
        ticker = row["ETF"]
        z_score = row["z-score"]
        fe = row[f"Free Energy ({selected_window}d)"]
        action = row[f"Action @ {selected_window}d"]
        rank_badge = "⭐" if idx == 0 else ("🥈" if idx == 1 else "🥉")
        
        with cols[idx]:
            st.markdown(f"""
<div class="top-card">
  <div class="ticker">{rank_badge} {ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">Free Energy = {fe:.3f}</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

    # ── Legend ──────────────────────────────────────────────────────────────────
    with st.expander("🎨 Color Legend", expanded=False):
        st.markdown("""
| Color | Meaning |
|-------|---------|
| 🟢 **Green** | **BUY** — Favorable risk/reward, low free energy |
| 🟡 **Yellow** | **HOLD** — Neutral, wait for confirmation |
| 🟠 **Orange** | **REDUCE** — Unfavorable risk/reward |
| 🔴 **Red** | **SELL** — High free energy, avoid |
| ⭐ **Blue** | **TOP PICK** — Best risk/reward in the universe |
        """)

    st.caption(f"Run date: {data2.get('run_date','?')} · Green = Best to Buy · Red = Worst to Buy")
