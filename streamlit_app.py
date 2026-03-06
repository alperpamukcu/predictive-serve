# streamlit_app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import textwrap

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from joblib import load

from src.utils.config import PROCESSED_DIR, MODELS_DIR
from src.utils.feature_utils import load_feature_list

try:
    from src.analysis.metrics import compute_overall_metrics  # type: ignore
except Exception:
    compute_overall_metrics = None

from src.predict.whatif import build_feature_row  # type: ignore


# =========================
# Page Config
# =========================
st.set_page_config(page_title="Predictive Serve", layout="wide")


# =========================
# i18n
# =========================
TR = {
    "title": "Predictive Serve",
    "settings": "Ayarlar",
    "language": "Dil",
    "dev_mode": "Geliştirici modu",
    "reset": "Filtreleri sıfırla",
    "filters_global": "Genel Filtreler",
    "filters_matches": "Match Filters",
    "date_range": "Tarih aralığı",
    "date_single_hint": "Tek tarih seçtiniz. Aynı günü başlangıç-bitiş olarak kabul ettim.",
    "surface": "Zemin",
    "metrics": "Seçili Dönem Metrikleri",
    "model_logloss": "Güven Kalitesi (Düşük iyi)",
    "model_brier": "Hata Puanı (Düşük iyi)",
    "model_acc": "Kazananı Bilme (%)",
    "market_logloss": "Market Güven K.",
    "market_brier": "Market Hata P.",
    "market_acc": "Market Bilme (%)",
    "tabs_matches": "Matches",
    "tabs_whatif": "What-if",
    "tabs_leaderboard": "Leaderboard",
    "player_pick": "Oyuncu (yazıp ara)",
    "player_any": "Hepsi",
    "tournament_search": "Turnuva (içeren)",
    "only_market": "Sadece market olasılığı olanlar",
    "only_model": "Sadece model olasılığı olanlar",
    "min_edge": "Minimum Avantaj (Fırsat)",
    "found": "Bulunan maç sayısı",
    "sort_by": "Sırala",
    "ascending": "Artan",
    "rows": "Satır",
    "select_info": "Bir satır seç → maç detayı aşağıda açılır.",
    "match_summary": "Maç Özeti",
    "model_vs_market": "Model vs Market",
    "whatif_title": "What-if",
    "player_a": "Oyuncu A",
    "player_b": "Oyuncu B",
    "match_date": "Maç tarihi",
    "round_code": "Round kodu (opsiyonel)",
    "round_any": "Seçilmedi",
    "enter_odds": "Oran gir (opsiyonel)",
    "odds_a": "Odds A",
    "odds_b": "Odds B",
    "snap_title": "Simulation Settings (Time Travel)",
    "snap_help": "Bu tarihteki oyuncu formunu ve Elo puanını baz alır (Geçmişe gitme simülasyonu).",
    "snap_current": "Current (No Time Travel)",
    "sim_settings": "Time Machine (Optional)",
    "calc": "✅ HESAPLA",
    "prediction": "Tahmin",
    "model_picks": "Modelin seçimi",
    "confidence": "Güven",
    "low": "Düşük",
    "medium": "Orta",
    "high": "Yüksek",
    "leaderboard_title": "Leaderboard",
    "min_matches": "Min maç sayısı",
    "metric": "Metrik",
    "top_n": "Top N",
    "winrate": "Win rate",
    "wins": "Wins",
    "avg_edge": "Avg edge (varsa)",
    "players_to_plot": "Grafikte gösterilecek oyuncular",
}
EN = {
    "title": "Predictive Serve",
    "settings": "Settings",
    "language": "Language",
    "dev_mode": "Developer mode",
    "reset": "Reset filters",
    "filters_global": "Global Filters",
    "filters_matches": "Match Filters",
    "date_range": "Date range",
    "date_single_hint": "You picked a single date. Using it as both start and end.",
    "surface": "Surface",
    "metrics": "Overall Performance",
    "model_logloss": "Confidence Quality (Lower is Better)",
    "model_brier": "Error Score (Lower is Better)",
    "model_acc": "Winner Prediction (%)",
    "market_logloss": "Market Confidence",
    "market_brier": "Market Error",
    "market_acc": "Market Accuracy",
    "tabs_matches": "Matches Explorer",
    "tabs_whatif": "Predictor (What-if)",
    "tabs_leaderboard": "Player Rankings",
    "player_pick": "Filter by Player",
    "player_any": "All Players",
    "tournament_search": "Filter by Tournament",
    "only_market": "Has Market Odds",
    "only_model": "Has AI Prediction",
    "min_edge": "Min Value (Ai Advantage)",
    "found": "Matches found",
    "sort_by": "Sort by",
    "ascending": "Ascending",
    "rows": "Rows",
    "select_info": "Select a match above to see the AI analysis card.",
    "match_summary": "Match Analysis Card",
    "model_vs_market": "AI vs Market",
    "whatif_title": "Match Predictor",
    "player_a": "Player A",
    "player_b": "Player B",
    "match_date": "Match Date",
    "round_code": "Round (Optional)",
    "round_any": "Any / None",
    "enter_odds": "Input Market Odds (Optional)",
    "odds_a": "Odds for A",
    "odds_b": "Odds for B",
    "snap_title": "Simulation Settings (Time Travel)",
    "snap_help": "Simulate match as if it happened on this date (uses historical form/Elo).",
    "snap_current": "Current (No Time Travel)",
    "sim_settings": "Time Machine (Optional)",
    "calc": "🔮 PREDICT MATCH",
    "prediction": "Prediction Result",
    "model_picks": "AI Picks",
    "confidence": "Confidence",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "leaderboard_title": "Player Rankings",
    "min_matches": "Min Matches Played",
    "metric": "Ranking Metric",
    "top_n": "Show Top",
    "winrate": "Win Rate",
    "wins": "Total Wins",
    "avg_edge": "Avg Value (AI Advantage)",
    "players_to_plot": "Players to plot",
}

def T(lang: str) -> Dict[str, str]:
    return TR if lang == "TR" else EN


# =========================
# CSS (single dark glass)
# =========================

# =========================
# CSS & Styling
# =========================
def get_css() -> str:
    return """
    <style>
    /* Global Clean Up */
    header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
    div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }
    #MainMenu, footer { visibility: hidden; }
    
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem;
        max_width: 1200px;
    }

    /* Modern Dark Theme Background */
    .stApp {
        background-color: #0e1117;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(255, 60, 0, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 100% 0%, rgba(0, 100, 255, 0.05) 0%, transparent 40%);
        color: #f0f2f6;
    }

    /* Cards */
    .ps-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ps-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .ps-title { 
        font-size: 1.25rem; 
        font-weight: 700; 
        margin-bottom: 8px; 
        color: #fff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .ps-metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .ps-metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
    }

    /* Progress Bars in Tape */
    .tape-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 4px;
    }
    .tape-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .bar-a { background: linear-gradient(90deg, #ff4b4b, #ff8f8f); }
    .bar-b { background: linear-gradient(90deg, #4b9cff, #8fbfff); }
    
    /* Buttons */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    
    </style>
    """

def inject_css():
    st.markdown(get_css(), unsafe_allow_html=True)


# =========================
# Paths
# =========================
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.txt"

PRED_CANDIDATES = [
    PROCESSED_DIR / "all_predictions.csv",
    PROCESSED_DIR / "val_predictions.csv",
    PROCESSED_DIR / "predictions.csv",
]
HISTORY_CANDIDATES = [
    PROCESSED_DIR / "matches_with_elo_form_sets.csv",
    PROCESSED_DIR / "train_dataset.csv",
    PROCESSED_DIR / "matches_with_features.csv",
]


# =========================
# Utils
# =========================
def clean_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    return " ".join(s.split())

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col in df.columns:
        vals = df[col].dropna().astype(str).map(clean_text)
        vals = vals[vals != ""]
        return sorted(vals.unique().tolist())
    return []

def parse_date_range(val, t: Dict[str, str]) -> Tuple[Optional[Any], Optional[Any], bool]:
    if val is None:
        return None, None, False
    if isinstance(val, (tuple, list)):
        if len(val) == 2:
            return val[0], val[1], False
        if len(val) == 1:
            return val[0], val[0], True
        return None, None, False
    return val, val, True

def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})

    if "player_1" in df.columns and "playerA" not in df.columns:
        df = df.rename(columns={"player_1": "playerA"})
    if "player_2" in df.columns and "playerB" not in df.columns:
        df = df.rename(columns={"player_2": "playerB"})

    for c in ["playerA", "playerB", "surface", "tournament", "round"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)

    if "pA" in df.columns and "p_model" not in df.columns:
        df = df.rename(columns={"pA": "p_model"})
    if "p_market" in df.columns and "pA_market" not in df.columns:
        df = df.rename(columns={"p_market": "pA_market"})

    if "p_model" in df.columns and "pA_market" in df.columns and "edge" not in df.columns:
        df["edge"] = df["p_model"] - df["pA_market"]

    if "y" in df.columns and "winner" not in df.columns and "playerA" in df.columns and "playerB" in df.columns:
        df["winner"] = np.where(df["y"].astype(int) == 1, df["playerA"], df["playerB"])
    return df

def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})

    if "player_1" in df.columns and "playerA" not in df.columns:
        df = df.rename(columns={"player_1": "playerA"})
    if "player_2" in df.columns and "playerB" not in df.columns:
        df = df.rename(columns={"player_2": "playerB"})

    for c in ["playerA", "playerB", "surface", "tournament", "round"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> Tuple[pd.DataFrame, str]:
    path = first_existing(PRED_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Predictions not found.")
    df = pd.read_csv(path)
    df = normalize_predictions(df)

    # ENRICHMENT: Merge with history to get tournament/round if missing
    # predictions often lack these columns, but history has them.
    if "tournament" not in df.columns or df["tournament"].isna().all():
        hist_path = first_existing(HISTORY_CANDIDATES)
        if hist_path and hist_path.exists():
            hdf = pd.read_csv(hist_path)
            hdf = normalize_history(hdf)
            
            # Use a subset to merge
            cols_to_merge = ["date", "playerA", "playerB"]
            cols_to_pull = ["tournament", "round"]
            
            # Check availability
            av_cols = [c for c in cols_to_pull if c in hdf.columns and c not in df.columns]
            
            if av_cols:
                # Deduplicate history just in case
                hdf = hdf.drop_duplicates(subset=cols_to_merge)
                
                # DEBUG: Check overlap
                # common = df.merge(hdf[cols_to_merge], on=cols_to_merge, how="inner")
                # print(f"DEBUG: Common rows: {len(common)} / {len(df)}")
                
                df = df.merge(hdf[cols_to_merge + av_cols], on=cols_to_merge, how="left")
                
                # Check match success
                nulls = df["tournament"].isna().sum()
                print(f"DEBUG: Tournament nulls after merge: {nulls} / {len(df)}")

    return df, str(path)

@st.cache_data(show_spinner=False)
def load_history() -> Tuple[pd.DataFrame, str]:
    path = first_existing(HISTORY_CANDIDATES)
    if path is None:
        raise FileNotFoundError("History dataset not found.")
    df = pd.read_csv(path)
    df = normalize_history(df)
    return df, str(path)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load(MODEL_PATH)
    imputer = load(IMPUTER_PATH)
    try:
        feature_cols = load_feature_list(FEATURE_COLS_PATH)  # type: ignore
    except TypeError:
        feature_cols = load_feature_list(str(FEATURE_COLS_PATH))  # type: ignore
    return model, imputer, feature_cols


def apply_global_filters(df: pd.DataFrame, d1, d2, surfaces: List[str]) -> pd.DataFrame:
    out = df.copy()
    if d1 and d2 and "date" in out.columns:
        out = out[(out["date"].dt.date >= d1) & (out["date"].dt.date <= d2)]
    if surfaces and "surface" in out.columns:
        out = out[out["surface"].isin(surfaces)]
    return out


def apply_match_filters(df: pd.DataFrame, player_pick: str, tournaments: List[str],
                        only_market: bool, only_model: bool, min_edge: Optional[float]) -> pd.DataFrame:
    out = df.copy()

    if player_pick and player_pick != "__ALL__":
        a = out.get("playerA", pd.Series("", index=out.index)).astype(str)
        b = out.get("playerB", pd.Series("", index=out.index)).astype(str)
        out = out[(a == player_pick) | (b == player_pick)]

    if tournaments and "tournament" in out.columns:
        out = out[out["tournament"].isin(tournaments)]

    if only_market and "pA_market" in out.columns:
        out = out[out["pA_market"].notna()]
    if only_model and "p_model" in out.columns:
        out = out[out["p_model"].notna()]

    if (min_edge is not None) and ("edge" in out.columns):
        out = out[out["edge"] >= min_edge]

    return out


def confidence_label(pA: float, t: Dict[str, str]) -> str:
    d = abs(pA - 0.5)
    if d < 0.05:
        return t["low"]
    if d < 0.12:
        return t["medium"]
    return t["high"]

def predict_from_row(model, imputer, feature_cols: List[str], row_df: pd.DataFrame) -> float:
    X = row_df[feature_cols].copy()
    X_imp = imputer.transform(X)
    return float(model.predict_proba(X_imp)[0, 1])


def render_kpis(df: pd.DataFrame, t: Dict[str, str]):
    st.markdown(f"### {t['metrics']}")
    if compute_overall_metrics is None:
        st.info("Metrics module not available.")
        return
    if df.empty:
        st.warning("No data for selected filters.")
        return
    m = compute_overall_metrics(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(t["model_logloss"], f"{m.model_logloss:.4f}", help="LogLoss: Hata cezalandırma puanıdır. 0'a ne kadar yakınsa, model o kadar 'emin' ve 'doğru'dur. (Düşük = İyi)")
    with c2:
        st.metric(t["model_brier"], f"{m.model_brier:.4f}", help="Brier Score: Tahmin sapmasıdır. 0 = Mükemmel tahmin. (Düşük = İyi)")
    with c3:
        st.metric(t["model_acc"], f"{m.model_acc:.3f}", help="Sadece kazananı bilme oranıdır. (Yüksek = İyi)")
    with c4:
        st.metric(t["market_logloss"], "—" if m.market_logloss is None else f"{m.market_logloss:.4f}", help="Marketin (Bahis bürolarının) hata puanı.")
    with c5:
        st.metric(t["market_brier"], "—" if m.market_brier is None else f"{m.market_brier:.4f}")
    with c6:
        st.metric(t["market_acc"], "—" if m.market_acc is None else f"{m.market_acc:.3f}")




# =========================
# Visualizations
# =========================
def render_match_card(row: pd.Series, t: Dict[str, str]):
    """Renders a beautiful card for a historical match."""
    
    # Extract data securely
    def val(k): return row[k] if k in row.index else None
    
    date = pd.to_datetime(val("date")).strftime("%Y-%m-%d") if val("date") else "Unknown Date"
    tourn = val("tournament") or "Unknown Tournament"
    surface = val("surface") or "Unknown Surface"
    round_ = val("round") or ""
    
    pA = val("p_model")
    pMarket = val("pA_market")
    edge = val("edge")
    
    winner = val("winner")
    playerA = val("playerA")
    playerB = val("playerB")
    
    # --- Explainability Logic (Key Factors) ---
    factors = []
    
    # helper for safe floats
    def fgap(k_a, k_b):
        va = val(k_a) if val(k_a) else 0
        vb = val(k_b) if val(k_b) else 0
        return va - vb

    # 1. Elo Diff
    elo_d = fgap("eloA", "eloB")
    if abs(elo_d) > 20:
        adv = playerA if elo_d > 0 else playerB
        factors.append(f"🏆 <b>Genel Güç (Elo):</b> {adv}, rakibinden daha yüksek puana sahip fark ({abs(elo_d):.0f}).")
    
    # 2. Surface Elo Diff
    surf_d = fgap("elo_surfaceA", "elo_surfaceB")
    if abs(surf_d) > 20:
        adv = playerA if surf_d > 0 else playerB
        if adv != (playerA if elo_d > 0 else playerB): 
             factors.append(f"🎾 <b>Zemin Uyumu ({surface}):</b> {adv}, bu zeminde daha başarılı ({abs(surf_d):.0f} puan fark).")

    # 3. Form Diff (Last 5 matches winrate)
    form_d = fgap("form_winrateA_5", "form_winrateB_5")
    if abs(form_d) > 0.1: # 10% diff
        adv = playerA if form_d > 0 else playerB
        factors.append(f"🔥 <b>Form Durumu:</b> {adv} son maçlarda daha formda (+%{abs(form_d)*100:.0f}).")
        
    factors_html = ""
    if factors:
         lis = "".join([f"<li style='margin-bottom:4px;'>{f}</li>" for f in factors])
         factors_html = f"""
         <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 0.85rem;">
            <div style="opacity: 0.7; font-weight: bold; margin-bottom: 6px;">💡 Neden bu tahmin? (Öne Çıkanlar)</div>
            <ul style="padding-left: 20px; opacity: 0.8;">{lis}</ul>
         </div>
         """

    # --- Consolidated Metrics Logic ---
    # 1. Determine who is predicted (A if p>=0.5 else B)
    pred_is_A = (pA >= 0.5)
    pred_player = playerA if pred_is_A else playerB
    
    # 2. Align metrics to the Predicted Player
    
    # AI Confidence (Flip if predicting B)
    display_prob = pA if pred_is_A else (1.0 - pA)
    display_prob_str = f"{display_prob*100:.1f}%" if pd.notna(display_prob) else "—"

    # Market Odds (Flip if predicting B)
    pMarket_val = pMarket
    if pMarket_val is not None:
        if abs(pMarket_val) <= 1.0:
            if not pred_is_A: pMarket_val = 1.0 - pMarket_val
            pMarket_val *= 100 # scale to %
        else:
            # Already %? (e.g. 34.0)
            if not pred_is_A: pMarket_val = 100.0 - pMarket_val
    pMarket_fmt = f"{pMarket_val:.1f}%" if pd.notna(pMarket_val) else "—"

    # Edge (Flip SIGN if predicting B)
    edge_val = edge
    if edge_val is not None:
        # Edge is defined as pModel - pMarket.
        # If we flipped both pModel and pMarket (1-p), the diff becomes (1-pA) - (1-pM) = pM - pA = - (pA - pM) = -Edge.
        if not pred_is_A: edge_val = -1.0 * edge_val
        if abs(edge_val) <= 1.0: edge_val *= 100 # scale to %
    edge_fmt = f"{edge_val:+.1f}%" if pd.notna(edge_val) else "—"

    html_content = f"""<div class="ps-card"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; opacity: 0.7; font-size: 0.9rem;"><span>📅 {date}</span><span>🏆 {tourn} ({surface}) {round_}</span></div><div style="display: flex; align-items: center; justify: space-around; margin: 20px 0;"><div style="text-align: center;"><div style="font-size: 1.8rem; font-weight: {800 if winner == playerA else 400}; color: {'#ff4b4b' if winner == playerA else 'inherit'};">{playerA}</div></div><div style="font-size: 1.2rem; font-weight: bold; opacity: 0.5;">VS</div><div style="text-align: center;"><div style="font-size: 1.8rem; font-weight: {800 if winner == playerB else 400}; color: {'#4b9cff' if winner == playerB else 'inherit'};">{playerB}</div></div></div><div style="margin-top: 25px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;"><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;"><div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 12px;"><div class="ps-metric-label">🤖 AI Prediction</div><div class="ps-metric-val">{display_prob_str}</div><div style="font-size: 0.8rem; opacity: 0.6;">for {pred_player}</div></div><div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 12px;"><div class="ps-metric-label">🏦 Market Odds</div><div class="ps-metric-val">{pMarket_fmt}</div><div style="font-size: 0.8rem; opacity: 0.6;">Implied Prob.</div></div><div style="background: {'rgba(40, 200, 64, 0.15)' if (edge_val and edge_val > 0) else 'rgba(255,255,255,0.05)'}; padding: 10px; border-radius: 12px; border: { '1px solid #4ade80' if (edge_val and edge_val > 5.0) else 'none'};"><div class="ps-metric-label">💎 Value (Edge)</div><div class="ps-metric-val" style="color: {'#4ade80' if (edge_val and edge_val > 0) else 'inherit'};">{edge_fmt}</div><div style="font-size: 0.8rem; opacity: {1 if (edge_val and edge_val > 5.0) else 0.6}; color: {'#4ade80' if (edge_val and edge_val > 5.0) else 'inherit'};">{ "🌟 GOOD BET" if (edge_val and edge_val > 5.0) else "No Value" }</div></div></div>{factors_html}</div></div>"""
    
    st.markdown(html_content, unsafe_allow_html=True)

def render_tale_of_the_tape(row_df: pd.DataFrame, playerA: str, playerB: str, surface: str):
    """Shows a side-by-side comparison of key stats."""
    
    def get_val(col):
        if col in row_df.columns and pd.notna(row_df[col].iloc[0]):
            return row_df[col].iloc[0]
        return 0

    eloA = get_val("eloA")
    eloB = get_val("eloB")
    surfA = get_val("elo_surfaceA")
    surfB = get_val("elo_surfaceB")
    formA = get_val("form_winrateA_5") * 100
    formB = get_val("form_winrateB_5") * 100
    h2hA = get_val("h2h_winrateA") * 100
    h2hB = get_val("h2h_winrateB") * 100
    
    st.markdown(f"<div class='ps-title'>🥊 Tale of the Tape</div>", unsafe_allow_html=True)
    
    # Helper to render a single row
    def render_row(label, valA, valB, fmt="{:.0f}", is_percent=False):
        c1, c2, c3 = st.columns([1, 2, 1])
        
        # Color logic: Highlight winner
        colorA = "#ff4b4b" if valA > valB else "rgba(255,255,255,0.6)"
        colorB = "#4b9cff" if valB > valA else "rgba(255,255,255,0.6)"
        weightA = "700" if valA > valB else "400"
        weightB = "700" if valB > valA else "400"

        with c1:
            st.markdown(
                f"<div style='text-align: right; color: {colorA}; font-weight: {weightA}; font-size: 1.1rem;'>"
                f"{fmt.format(valA)}"
                f"</div>", 
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"<div style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem; text-transform: uppercase;'>{label}</div>", 
                unsafe_allow_html=True
            )
            # Mini bar chart
            total = valA + valB if (valA + valB) > 0 else 1
            pctA = (valA / total) * 100
            
            st.markdown(
                f"""
                <div style="display: flex; height: 6px; width: 100%; border-radius: 3px; overflow: hidden; background: rgba(255,255,255,0.1);">
                    <div style="width: {pctA}%; background: #ff4b4b;"></div>
                    <div style="width: {100-pctA}%; background: #4b9cff;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"<div style='text-align: left; color: {colorB}; font-weight: {weightB}; font-size: 1.1rem;'>"
                f"{fmt.format(valB)}"
                f"</div>", 
                unsafe_allow_html=True
            )
    
    render_row("Overall Rating (Elo)", eloA, eloB)
    render_row(f"{surface} Rating", surfA, surfB)
    render_row("Last 5 Form (%)", formA, formB, fmt="{:.1f}%")
    render_row("H2H Win (%)", h2hA, h2hB, fmt="{:.1f}%")


def plot_elo_history(playerA: str, playerB: str, history_df: pd.DataFrame, date_limit="2020-01-01"):
    """Plots Elo history for both players."""
    st.markdown("<div class='ps-title'>📈 Elo History (Since 2020)</div>", unsafe_allow_html=True)
    
    # Filter for efficiency
    mask = (
        (history_df["date"] >= date_limit) & 
        (
            (history_df["playerA"].isin([playerA, playerB])) | 
            (history_df["playerB"].isin([playerA, playerB]))
        )
    )
    df = history_df[mask].copy()
    
    if df.empty:
        st.caption("No history data found for these players since 2020.")
        return

    # Extract Elo at each match date for each player
    # A player can be playerA or playerB
    
    data_list = []
    
    # Only sort once
    df = df.sort_values("date")

    for p in [playerA, playerB]:
        # Matches where p is A
        asA = df[df["playerA"] == p][["date", "eloA"]].rename(columns={"eloA": "elo"})
        # Matches where p is B
        asB = df[df["playerB"] == p][["date", "eloB"]].rename(columns={"eloB": "elo"})
        
        combined = pd.concat([asA, asB]).sort_values("date")
        combined["Player"] = p
        data_list.append(combined)

    if not data_list:
        return
        
    final_df = pd.concat(data_list)
    
    # Check for missing players and warn
    present_players = final_df["Player"].unique()
    missing = set([playerA, playerB]) - set(present_players)
    if missing:
        for m in missing:
            st.warning(f"⚠️ No match history found for {m} since 2020. Chart line will be missing.")
        
        # Fallback: Career Summary Table if chart is empty/partial
        if len(missing) == 2 or final_df.empty:
             # Calculate simple stats from history_df manually here
             stats = []
             for p in [playerA, playerB]:
                 p_matches = history_df[(history_df["playerA"] == p) | (history_df["playerB"] == p)]
                 wins = p_matches[p_matches["winner"] == p].shape[0]
                 total = p_matches.shape[0]
                 wr = (wins / total * 100) if total > 0 else 0.0
                 stats.append({"Player": p, "Matches": total, "Wins": wins, "Win Rate": f"{wr:.1f}%"})
             
             st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)
             return

    # pivot for st.line_chart types: index=date, columns=Player, values=elo
    # Instead of pivot, let's keep it long format for Altair
    # final_df has columns: date, elo, elo_surface, Player
    
    # Altair Chart
    base = alt.Chart(final_df).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('elo:Q', title='Elo Rating', scale=alt.Scale(zero=False)),
        color=alt.Color('Player:N', scale=alt.Scale(domain=[playerA, playerB], range=['#ff4b4b', '#4b9cff'])),
        tooltip=['date:T', 'Player', 'elo']
    )

    chart = base.mark_line(opacity=0.8, strokeWidth=3) + base.mark_circle(size=60, opacity=1.0)
    
    st.altair_chart(chart, use_container_width=True)

    # Add Career Summary Below Chart (Customer Request)
    st.markdown("#### Call-Time Career Summary")
    stats = []
    for p in [playerA, playerB]:
        p_matches = history_df[(history_df["playerA"] == p) | (history_df["playerB"] == p)]
        wins_as_A = p_matches[(p_matches["playerA"] == p) & (p_matches["winner"] == "A")].shape[0]
        wins_as_B = p_matches[(p_matches["playerB"] == p) & (p_matches["winner"] == "B")].shape[0]
        wins = wins_as_A + wins_as_B
        t_matches = p_matches.shape[0]
        wr = (wins / t_matches * 100) if t_matches > 0 else 0.0
        stats.append({"Player": p, "Matches": t_matches, "Wins": wins, "Win Rate": f"{wr:.1f}%"})
    
    st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)


# =========================
# UI Boot
# =========================
inject_css()

# Load base predictions once
pred_df, pred_path = load_predictions()

# Sidebar Settings
st.sidebar.subheader("⚙️")
lang = st.sidebar.selectbox("Language / Dil", ["EN", "TR"], index=1, key="lang")
t = T(lang)
dev_mode = st.sidebar.checkbox(t["dev_mode"], value=False, key="dev_mode")

# Reset button (clears filters + rerun)
if st.sidebar.button(t["reset"], use_container_width=True):
    st.cache_data.clear() # FORCE CLEAR CACHE
    for k in [
        "g_dates", "g_surfaces",
        "m_player", "m_tourn", "m_only_market", "m_only_model", "m_min_edge",
        "lb_min_matches", "lb_metric", "lb_topn",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Global filters (affect KPIs + all tabs)
st.sidebar.subheader(t["filters_global"])
if "date" in pred_df.columns and pred_df["date"].notna().any():
    min_d = pred_df["date"].min().date()
    max_d = pred_df["date"].max().date()
    dr = st.sidebar.date_input(t["date_range"], value=(min_d, max_d), key="g_dates")
    g_d1, g_d2, single = parse_date_range(dr, t)
    if single:
        st.sidebar.caption(t["date_single_hint"])
else:
    g_d1 = g_d2 = None

all_surfaces = safe_unique(pred_df, "surface")
g_surfaces = st.sidebar.multiselect(t["surface"], all_surfaces, default=all_surfaces, key="g_surfaces")

# Apply global filters for KPIs
filtered_for_kpi = apply_global_filters(pred_df, g_d1, g_d2, g_surfaces)

# Header + KPIs (NOW FILTERED)
st.markdown(f"<div class='ps-title'>{t['title']}</div>", unsafe_allow_html=True)
render_kpis(filtered_for_kpi, t)

if dev_mode:
    st.caption(f"[DEV] predictions source: {pred_path} | rows(all)={len(pred_df):,} rows(filtered)={len(filtered_for_kpi):,}")
    st.caption(f"[DEV] columns: {', '.join(list(pred_df.columns)[:30])}{' ...' if len(pred_df.columns)>30 else ''}")

st.divider()

tab_matches, tab_whatif, tab_leaderboard = st.tabs([t["tabs_matches"], t["tabs_whatif"], t["tabs_leaderboard"]])


# =========================
# Matches tab
# =========================
with tab_matches:
    st.markdown("<div class='ps-card'><div class='ps-title'>Matches Explorer</div></div>", unsafe_allow_html=True)

    st.sidebar.subheader(t["filters_matches"])

    # Player selectbox = type-to-search (solves "Fed" issue)
    players_all = sorted(pd.unique(pd.concat([pred_df.get("playerA", pd.Series(dtype=str)),
                                              pred_df.get("playerB", pd.Series(dtype=str))], ignore_index=True)).astype(str))
    players_all = [p for p in (clean_text(x) for x in players_all) if p]
    player_opts = [t["player_any"]] + players_all

    player_pick = st.sidebar.selectbox(t["player_pick"], player_opts, index=0, key="m_player")
    player_pick_internal = "__ALL__" if player_pick == t["player_any"] else player_pick

    tourn_opts = safe_unique(pred_df, "tournament")
    tournaments = st.sidebar.multiselect(t["tournament_search"], tourn_opts, key="m_tourn")

    only_market = st.sidebar.checkbox(t["only_market"], value=False, key="m_only_market")
    only_model = st.sidebar.checkbox(t["only_model"], value=True, key="m_only_model")

    min_edge = None
    if "edge" in pred_df.columns:
        min_edge = st.sidebar.slider(t["min_edge"], -1.0, 1.0, 0.0, 0.01, key="m_min_edge")

    # Apply global + match filters
    base = apply_global_filters(pred_df, g_d1, g_d2, g_surfaces)
    fdf = apply_match_filters(base, player_pick_internal, tournaments, only_market, only_model, min_edge)

    st.write(f"{t['found']}: **{len(fdf):,}**")

    show_cols = [c for c in [
        "date", "tournament", "round", "surface",
        "playerA", "playerB", "winner",
        "p_model", "pA_market", "edge"
    ] if c in fdf.columns]

    display_cols = []
    
    # Translation map for Sort By dropdown
    sort_map = {
        "date": t["match_date"],
        "tournament": "Turnuva",
        "round": "Round",
        "surface": t["surface"],
        "playerA": t["player_a"],
        "playerB": t["player_b"],
        "winner": "Kazanan",
        "p_model": "AI Güven (Prob)",
        "pA_market": "Market Oranı",
        "edge": "Değer (Edge/Fırsat)"
    }
    
    # Reverse map to find original col from display name
    rev_sort_map = {v: k for k, v in sort_map.items()}

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # Show translated options
        # We filter show_cols to what exists, then map them
        available_sort_options = [sort_map.get(c, c) for c in show_cols]
        selected_disp = st.selectbox(t["sort_by"], options=available_sort_options)
        # Convert back to internal name
        sort_by = rev_sort_map.get(selected_disp, selected_disp)

    with c2:
        ascending = st.checkbox(t["ascending"], value=False)
    with c3:
        limit = st.selectbox(t["rows"], [100, 250, 500, 1000], index=1)

    if sort_by in fdf.columns:
        fdf = fdf.sort_values(sort_by, ascending=ascending)

    view = fdf[show_cols].head(limit).copy() if show_cols else fdf.head(limit).copy()

    # RENAME COLS FOR UI
    # RENAME COLS FOR UI
    # Scale to percentage for display (0.9 -> 90.0)
    # AND FLIP if p_model < 0.5 (Customer Request: Show Winner Probability, not Player A)
    if "p_model" in view.columns:
        # We need to act on rows where p_model < 0.5
        # 1) If p_model < 0.5 => It means Model predicts Player B.
        #    We want to show p(B) = 1 - p(A).
        #    We also must flip Market Odds to be for Player B => 1 - pMarket(A).
        #    Edge sign also flips.
        
        mask = view["p_model"] < 0.5
        view.loc[mask, "p_model"] = 1.0 - view.loc[mask, "p_model"]
        
        if "pA_market" in view.columns:
            view.loc[mask, "pA_market"] = 1.0 - view.loc[mask, "pA_market"]
            
        if "edge" in view.columns:
            view.loc[mask, "edge"] = -1.0 * view.loc[mask, "edge"]

        # Now scale all to 100
        view["p_model"] = view["p_model"] * 100
        if "pA_market" in view.columns: view["pA_market"] = view["pA_market"] * 100
        if "edge" in view.columns: view["edge"] = view["edge"] * 100

    view = view.rename(columns={
        "date": "Date",
        "tournament": "Tournament",
        "round": "Round",
        "surface": "Surface",
        "playerA": "Player A",
        "playerB": "Player B",
        "winner": "Winner",
        "p_model": "AI Confidence",
        "pA_market": "Market Odds",
        "edge": "Value"
    })
    
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        height=400,
        selection_mode="single-row",
        on_select="rerun",
        key="match_selection",
        column_config={
            "AI Confidence": st.column_config.NumberColumn(
                "AI Confidence",
                help="Probability that Player A wins",
                format="%.1f%%"
            ),
            "Market Odds": st.column_config.NumberColumn(
                "Market Odds",
                help="Implied probability from bookmakers",
                format="%.1f%%"
            ),
            "Value": st.column_config.NumberColumn(
                "Value",
                help="Difference between AI model and Market",
                format="%.1f%%"
            )
        }
    )

    selected_rows = st.session_state.match_selection.get("selection", {}).get("rows", [])
    st.divider()

    if not selected_rows:
        st.info(t["select_info"])
    else:
        # Get the actual row index from the view
        selected_idx_loc = selected_rows[0]
        # RETRIEVE ORIGINAL DATA directly from fdf using the index label from view
        # This gives us the unscaled values (0.75 instead of 75.0) which render_match_card expects.
        original_idx = view.index[selected_idx_loc]
        simulated_row = fdf.loc[original_idx]

        render_match_card(simulated_row, t)

    if dev_mode:
        st.caption(f"[DEV] match filters => player={player_pick} tourn={tournaments} only_market={only_market} only_model={only_model} min_edge={min_edge}")


# =========================
# What-if tab
# =========================
with tab_whatif:
    st.markdown(f"<div class='ps-card'><div class='ps-title'>{t['whatif_title']}</div></div>", unsafe_allow_html=True)

    history_df, history_path = load_history()
    model, imputer, feature_cols = load_artifacts()

    players = sorted(pd.unique(pd.concat([history_df["playerA"], history_df["playerB"]], ignore_index=True)).astype(str))
    players = [p for p in (clean_text(x) for x in players) if p]

    surfaces = safe_unique(history_df, "surface") or ["Hard", "Clay", "Grass", "Carpet"]
    rounds = safe_unique(history_df, "round")
    round_options = [t["round_any"]] + rounds

    # odds toggle outside form
    use_odds = st.checkbox(t["enter_odds"], value=False)

    # SIMULATION SETTINGS (Outside Expander for visibility or inside as originally planned, but NO FORM)
    # Removing st.form to allow interactive check/uncheck behavior
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        # Player selection logic (using existing variables)
        pass # The player selection is above this block in the file, we assume it's fine.
             # We just need to confirm we aren't breaking indentation of lines 876-930 if we remove 'with st.form'.
             # Actually, lines 876 'with st.form' wraps the whole block. I need to dedent EVERYTHING.
             
    # Since dedenting hundreds of lines is risky with replace, I will Close the form? 
    # Or I can just replace the 'with st.form' line with a dummy and 'submitted' line with a st.button.
    
    # Strategy: Replace 'with st.form("whatif_form"):' with a dummy and 'submitted' line with a st.button.
    if True: # Removed st.form
        c1, c2 = st.columns(2)
        with c1:
            # Session State for Random Button
            if "rand_idx_A" not in st.session_state: st.session_state.rand_idx_A = 0
            playerA = st.selectbox(t["player_a"], players, index=st.session_state.rand_idx_A)
        with c2:
            if "rand_idx_B" not in st.session_state: st.session_state.rand_idx_B = 1 if len(players) > 1 else 0
            playerB = st.selectbox(t["player_b"], players, index=st.session_state.rand_idx_B)

        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            match_date = st.date_input(t["match_date"], value=pd.Timestamp.today().date())
        with r2:
            surface = st.selectbox(t["surface"], surfaces, index=0)
        with r3:
            round_choice = st.selectbox(t["round_code"], round_options, index=0)

        # Random Match Button (Small UI improvement)
        if st.button("🎲 Random Match Example"):
             import random
             if len(players) > 2:
                 idxA, idxB = random.sample(range(len(players)), 2)
                 st.session_state.rand_idx_A = idxA
                 st.session_state.rand_idx_B = idxB
                 st.rerun()

        oddsA = oddsB = None
        if use_odds:
            o1, o2 = st.columns(2)
            with o1:
                oddsA = st.number_input(t["odds_a"], min_value=1.01, value=2.0, step=0.01)
            with o2:
                oddsB = st.number_input(t["odds_b"], min_value=1.01, value=2.0, step=0.01)

        st.markdown(f"#### {t['snap_title']}")
        with st.expander(t["sim_settings"], expanded=False):
            st.info(f"ℹ️ {t['snap_help']}")
            
            c_snap1, c_snap2 = st.columns(2)
            with c_snap1:
                use_snapA = st.checkbox(f"Simulate Date for {playerA}", key="use_snapA")
                if use_snapA:
                    snap_dateA = st.date_input(f"Date for {playerA}", value=match_date, key="date_snapA")
                else:
                    snap_dateA = None
                    
            with c_snap2:
                use_snapB = st.checkbox(f"Simulate Date for {playerB}", key="use_snapB")
                if use_snapB:
                    snap_dateB = st.date_input(f"Date for {playerB}", value=match_date, key="date_snapB")
                else:
                    snap_dateB = None

        submitted = st.button(t["calc"], type="primary")

    if submitted:
        date_ts = pd.Timestamp(match_date)
        round_code = None if round_choice == t["round_any"] else round_choice

        row_df = None
        # No try/except - enforce new signature
        row_df = build_feature_row(
            history=history_df,
            feature_cols=feature_cols,
            playerA=playerA,
            playerB=playerB,
            surface=surface,
            date=date_ts,
            round_code=round_code,
            oddsA=float(oddsA) if oddsA is not None else None,
            oddsB=float(oddsB) if oddsB is not None else None,
            snapshot_dateA=pd.Timestamp(snap_dateA) if snap_dateA else None,
            snapshot_dateB=pd.Timestamp(snap_dateB) if snap_dateB else None,
        )

        pA = predict_from_row(model, imputer, feature_cols, row_df)
        pB = 1.0 - pA
        winner = playerA if pA >= 0.5 else playerB
        winner_prob = max(pA, pB)
        conf = confidence_label(pA, t)
        
        # --- NEW RESULT UI ---
        
        is_A_winner = (winner == playerA)
        # However, if players have same name, we use pA >= 0.5 logic
        if playerA == playerB:
             is_A_winner = (pA >= 0.5)

        # 1. Main Result Card
        st.markdown(f"""
        <div class="ps-card" style="text-align: center; border-left: 6px solid {'#ff4b4b' if is_A_winner else '#4b9cff'};">
            <div style="font-size: 1rem; opacity: 0.8;">PREDICTION</div>
            <div style="font-size: 2.2rem; font-weight: 800; margin: 10px 0;">
                <span style="color: {'#ff4b4b' if is_A_winner else '#ddd'}">{playerA} {'🏆' if is_A_winner else ''}</span>
                <span style="font-size: 1rem; vertical-align: middle; opacity: 0.5;">vs</span>
                <span style="color: {'#4b9cff' if not is_A_winner else '#ddd'}">{playerB} {'🏆' if not is_A_winner else ''}</span>
            </div>
            <div style="font-size: 1.2rem;">
                🏆 <b>{winner}</b> wins with <b>{winner_prob*100:.1f}%</b> probability
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                Confidence: <span style="color: yellow;">{conf}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Tale of the Tape & History
        col_vis1, col_vis2 = st.columns([1, 1])
        
        with col_vis1:
            st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
            render_tale_of_the_tape(row_df, playerA, playerB, surface)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_vis2:
            st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
            plot_elo_history(playerA, playerB, history_df)
            st.markdown("</div>", unsafe_allow_html=True)

        if dev_mode:
            st.caption(f"[DEV] history source: {history_path}")
            with st.expander("[DEV] feature row"):
                st.dataframe(row_df[feature_cols].T, use_container_width=True, height=450)


# =========================
# Leaderboard tab
# =========================
with tab_leaderboard:
    st.markdown(f"<div class='ps-card'><div class='ps-title'>{t['leaderboard_title']}</div></div>", unsafe_allow_html=True)

    if "y" not in pred_df.columns or "playerA" not in pred_df.columns or "playerB" not in pred_df.columns:
        st.error("Leaderboard requires columns: y, playerA, playerB (and date recommended).")
        st.stop()

    # Use global filters
    lb = apply_global_filters(pred_df, g_d1, g_d2, g_surfaces)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        min_matches = st.number_input(t["min_matches"], min_value=1, value=10, step=1, key="lb_min_matches")
    with c2:
        metric = st.selectbox(t["metric"], [t["winrate"], t["wins"], t["avg_edge"]], index=0, key="lb_metric")
    with c3:
        top_n = st.selectbox(t["top_n"], [10, 20, 50, 100], index=1, key="lb_topn")

    a = lb[["date", "playerA", "y"]].copy()
    a["player"] = a["playerA"]
    a["win"] = (a["y"].astype(int) == 1).astype(int)
    a["match"] = 1

    b = lb[["date", "playerB", "y"]].copy()
    b["player"] = b["playerB"]
    b["win"] = (b["y"].astype(int) == 0).astype(int)
    b["match"] = 1

    pl = pd.concat([a[["date", "player", "win", "match"]], b[["date", "player", "win", "match"]]], ignore_index=True)

    if "edge" in lb.columns:
        a2 = lb[["date", "playerA", "edge"]].copy()
        a2["player"] = a2["playerA"]
        a2["edge_player"] = a2["edge"]

        b2 = lb[["date", "playerB", "edge"]].copy()
        b2["player"] = b2["playerB"]
        b2["edge_player"] = -b2["edge"]

        ed = pd.concat([a2[["date", "player", "edge_player"]], b2[["date", "player", "edge_player"]]], ignore_index=True)
        pl = pl.merge(ed, on=["date", "player"], how="left")
    else:
        pl["edge_player"] = np.nan

    agg = pl.groupby("player", as_index=False).agg(
        matches=("match", "sum"),
        wins=("win", "sum"),
        avg_edge=("edge_player", "mean"),
    )
    agg = agg[agg["matches"] >= int(min_matches)]
    agg["win_rate"] = agg["wins"] / agg["matches"]

    if metric == t["winrate"]:
        agg = agg.sort_values(["win_rate", "matches"], ascending=[False, False])
    elif metric == t["wins"]:
        agg = agg.sort_values(["wins", "matches"], ascending=[False, False])
    else:
        if agg["avg_edge"].notna().any():
            agg = agg.sort_values(["avg_edge", "matches"], ascending=[False, False])
        else:
            agg = agg.sort_values(["win_rate", "matches"], ascending=[False, False])

    # Rename columns for UI
    agg = agg.rename(columns={
        "player": "Player",
        "matches": "Matches Played",
        "wins": "Wins",
        "win_rate": "Win Rate",
        "avg_edge": "Avg Value"
    })

    # Scale to percentage
    if "Win Rate" in agg.columns: agg["Win Rate"] = agg["Win Rate"] * 100
    if "Avg Value" in agg.columns: agg["Avg Value"] = agg["Avg Value"] * 100

    st.dataframe(
        agg.head(top_n), 
        use_container_width=True, 
        height=420,
        column_config={
            "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Value": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    if dev_mode:
        st.caption(f"[DEV] leaderboard rows after global filters: {len(lb):,}")
