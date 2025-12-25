# streamlit_app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from src.utils.config import PROCESSED_DIR, MODELS_DIR

try:
    from src.analysis.metrics import compute_overall_metrics  # type: ignore
except Exception:
    compute_overall_metrics = None

from src.predict.whatif import load_feature_list, build_feature_row  # type: ignore


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
    "model_logloss": "Model LogLoss",
    "model_brier": "Model Brier",
    "model_acc": "Model Doğruluk",
    "market_logloss": "Market LogLoss",
    "market_brier": "Market Brier",
    "market_acc": "Market Doğruluk",
    "tabs_matches": "Matches",
    "tabs_whatif": "What-if",
    "tabs_leaderboard": "Leaderboard",
    "player_pick": "Oyuncu (yazıp ara)",
    "player_any": "Hepsi",
    "tournament_search": "Turnuva (içeren)",
    "only_market": "Sadece market olasılığı olanlar",
    "only_model": "Sadece model olasılığı olanlar",
    "min_edge": "Min edge (model - market)",
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
    "snapshot_title": "Snapshot tarihleri (A ve B farklı olabilir)",
    "snap_a": "Snapshot tarihi A",
    "snap_b": "Snapshot tarihi B",
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
    "metrics": "Metrics for Selected Period",
    "model_logloss": "Model LogLoss",
    "model_brier": "Model Brier",
    "model_acc": "Model Acc",
    "market_logloss": "Market LogLoss",
    "market_brier": "Market Brier",
    "market_acc": "Market Acc",
    "tabs_matches": "Matches",
    "tabs_whatif": "What-if",
    "tabs_leaderboard": "Leaderboard",
    "player_pick": "Player (type to search)",
    "player_any": "All",
    "tournament_search": "Tournament (contains)",
    "only_market": "Only with market prob",
    "only_model": "Only with model prob",
    "min_edge": "Min edge (model - market)",
    "found": "Matches found",
    "sort_by": "Sort by",
    "ascending": "Ascending",
    "rows": "Rows",
    "select_info": "Select a row → match details appear below.",
    "match_summary": "Match Summary",
    "model_vs_market": "Model vs Market",
    "whatif_title": "What-if",
    "player_a": "Player A",
    "player_b": "Player B",
    "match_date": "Match date",
    "round_code": "Round code (optional)",
    "round_any": "Not selected",
    "enter_odds": "Enter odds (optional)",
    "odds_a": "Odds A",
    "odds_b": "Odds B",
    "snapshot_title": "Snapshot dates (A and B can be different)",
    "snap_a": "Snapshot date A",
    "snap_b": "Snapshot date B",
    "calc": "✅ CALCULATE",
    "prediction": "Prediction",
    "model_picks": "Model picks",
    "confidence": "Confidence",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "leaderboard_title": "Leaderboard",
    "min_matches": "Min matches",
    "metric": "Metric",
    "top_n": "Top N",
    "winrate": "Win rate",
    "wins": "Wins",
    "avg_edge": "Avg edge (if available)",
    "players_to_plot": "Players to plot",
}

def T(lang: str) -> Dict[str, str]:
    return TR if lang == "TR" else EN


# =========================
# CSS (single dark glass)
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
        div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .block-container { padding-top: 1.8rem; }

        .stApp {
          background:
            radial-gradient(1200px 600px at 20% 0%, rgba(255,0,120,0.10), transparent 55%),
            radial-gradient(900px 500px at 90% 10%, rgba(0,180,255,0.12), transparent 60%),
            radial-gradient(800px 500px at 50% 100%, rgba(120,255,120,0.08), transparent 60%),
            linear-gradient(180deg, #070a0f 0%, #070a0f 100%);
          color: #E6EDF3;
        }

        [data-testid="stSidebar"] {
          background: rgba(255,255,255,0.03);
          border-right: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(10px);
        }

        .ps-card {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 12px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
          backdrop-filter: blur(10px);
        }
        .ps-title { font-size: 1.35rem; font-weight: 800; margin: 0 0 6px 0; }
        .ps-sub { opacity: .85; margin: 0; }

        .stButton>button {
          border-radius: 14px !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          background: rgba(255,255,255,0.05) !important;
          box-shadow: 0 10px 18px rgba(0,0,0,0.35) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def apply_match_filters(df: pd.DataFrame, player_pick: str, tournament_q: str,
                        only_market: bool, only_model: bool, min_edge: Optional[float]) -> pd.DataFrame:
    out = df.copy()

    if player_pick and player_pick != "__ALL__":
        a = out.get("playerA", pd.Series("", index=out.index)).astype(str)
        b = out.get("playerB", pd.Series("", index=out.index)).astype(str)
        out = out[(a == player_pick) | (b == player_pick)]

    tq = clean_text(tournament_q).lower()
    if tq and "tournament" in out.columns:
        out = out[out["tournament"].astype(str).str.lower().str.contains(tq)]

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
        st.metric(t["model_logloss"], f"{m.model_logloss:.4f}")
    with c2:
        st.metric(t["model_brier"], f"{m.model_brier:.4f}")
    with c3:
        st.metric(t["model_acc"], f"{m.model_acc:.3f}")
    with c4:
        st.metric(t["market_logloss"], "—" if m.market_logloss is None else f"{m.market_logloss:.4f}")
    with c5:
        st.metric(t["market_brier"], "—" if m.market_brier is None else f"{m.market_brier:.4f}")
    with c6:
        st.metric(t["market_acc"], "—" if m.market_acc is None else f"{m.market_acc:.3f}")


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

    # Match-level filters in sidebar
    st.sidebar.subheader(t["filters_matches"])

    # Player selectbox = type-to-search (solves "Fed" issue)
    players_all = sorted(pd.unique(pd.concat([pred_df.get("playerA", pd.Series(dtype=str)),
                                              pred_df.get("playerB", pd.Series(dtype=str))], ignore_index=True)).astype(str))
    players_all = [p for p in (clean_text(x) for x in players_all) if p]
    player_opts = [t["player_any"]] + players_all

    player_pick = st.sidebar.selectbox(t["player_pick"], player_opts, index=0, key="m_player")
    player_pick_internal = "__ALL__" if player_pick == t["player_any"] else player_pick

    tournament_q = st.sidebar.text_input(t["tournament_search"], value="", key="m_tourn")

    only_market = st.sidebar.checkbox(t["only_market"], value=False, key="m_only_market")
    only_model = st.sidebar.checkbox(t["only_model"], value=True, key="m_only_model")

    min_edge = None
    if "edge" in pred_df.columns:
        min_edge = st.sidebar.slider(t["min_edge"], -1.0, 1.0, 0.0, 0.01, key="m_min_edge")

    # Apply global + match filters
    base = apply_global_filters(pred_df, g_d1, g_d2, g_surfaces)
    fdf = apply_match_filters(base, player_pick_internal, tournament_q, only_market, only_model, min_edge)

    st.write(f"{t['found']}: **{len(fdf):,}**")

    show_cols = [c for c in [
        "date", "tournament", "round", "surface",
        "playerA", "playerB", "winner",
        "p_model", "pA_market", "edge"
    ] if c in fdf.columns]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sort_by = st.selectbox(t["sort_by"], options=show_cols if show_cols else fdf.columns.tolist())
    with c2:
        ascending = st.checkbox(t["ascending"], value=False)
    with c3:
        limit = st.selectbox(t["rows"], [100, 250, 500, 1000], index=1)

    if sort_by in fdf.columns:
        fdf = fdf.sort_values(sort_by, ascending=ascending)

    view = fdf[show_cols].head(limit).copy() if show_cols else fdf.head(limit).copy()
    view.insert(0, "select", False)

    edited = st.data_editor(
        view,
        use_container_width=True,
        height=520,
        column_config={"select": st.column_config.CheckboxColumn("Select")},
        disabled=[c for c in view.columns if c != "select"],
    )

    selected_rows = edited[edited["select"] == True]
    st.divider()

    if selected_rows.empty:
        st.info(t["select_info"])
    else:
        row = selected_rows.iloc[0].drop(labels=["select"])
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"<div class='ps-card'><div class='ps-title'>{t['match_summary']}</div></div>", unsafe_allow_html=True)
            summary_keys = ["date", "tournament", "round", "surface", "playerA", "playerB", "winner"]
            st.json({k: (row[k] if k in row.index else None) for k in summary_keys})
        with right:
            st.markdown(f"<div class='ps-card'><div class='ps-title'>{t['model_vs_market']}</div></div>", unsafe_allow_html=True)
            if "p_model" in row.index and pd.notna(row["p_model"]):
                st.metric("p_model (A wins)", f"{float(row['p_model']):.3f}")
            if "pA_market" in row.index and pd.notna(row["pA_market"]):
                st.metric("p_market (A wins)", f"{float(row['pA_market']):.3f}")
            if "edge" in row.index and pd.notna(row["edge"]):
                st.metric("edge", f"{float(row['edge']):+.3f}")

    if dev_mode:
        st.caption(f"[DEV] match filters => player={player_pick} tourn='{tournament_q}' only_market={only_market} only_model={only_model} min_edge={min_edge}")


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

    with st.form("whatif_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            playerA = st.selectbox(t["player_a"], players, index=0)
        with c2:
            playerB = st.selectbox(t["player_b"], players, index=1 if len(players) > 1 else 0)

        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            match_date = st.date_input(t["match_date"], value=pd.Timestamp.today().date())
        with r2:
            surface = st.selectbox(t["surface"], surfaces, index=0)
        with r3:
            round_choice = st.selectbox(t["round_code"], round_options, index=0)

        oddsA = oddsB = None
        if use_odds:
            o1, o2 = st.columns(2)
            with o1:
                oddsA = st.number_input(t["odds_a"], min_value=1.01, value=2.0, step=0.01)
            with o2:
                oddsB = st.number_input(t["odds_b"], min_value=1.01, value=2.0, step=0.01)

        st.markdown(f"#### {t['snapshot_title']}")
        s1, s2 = st.columns(2)
        with s1:
            snapA_date = st.date_input(t["snap_a"], value=match_date, key="snapA_main")
        with s2:
            snapB_date = st.date_input(t["snap_b"], value=match_date, key="snapB_main")

        submitted = st.form_submit_button(t["calc"])

    if submitted:
        date_ts = pd.Timestamp(match_date)
        round_code = None if round_choice == t["round_any"] else round_choice

        row_df = None
        try:
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
                snapshot_dateA=pd.Timestamp(snapA_date),
                snapshot_dateB=pd.Timestamp(snapB_date),
            )
        except TypeError:
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
            )

        pA = predict_from_row(model, imputer, feature_cols, row_df)
        pB = 1.0 - pA
        winner = playerA if pA >= 0.5 else playerB
        winner_prob = max(pA, pB)
        conf = confidence_label(pA, t)

        st.markdown(
            f"<div class='ps-card'>"
            f"<div class='ps-title'>{t['prediction']}</div>"
            f"<p class='ps-sub'>{t['model_picks']}: <b>{winner}</b> "
            f"({winner_prob:.3f}) • {t['confidence']}: <b>{conf}</b></p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(pA, 0.0), 1.0), text=f"p({playerA} wins) = {pA:.3f} | p({playerB} wins) = {pB:.3f}")

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

    st.dataframe(agg.head(top_n), use_container_width=True, height=420)

    if dev_mode:
        st.caption(f"[DEV] leaderboard rows after global filters: {len(lb):,}")
