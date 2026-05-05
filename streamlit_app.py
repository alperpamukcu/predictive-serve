# streamlit_app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import html
import textwrap

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from joblib import load

from src.utils.config import PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT
from src.utils.feature_utils import load_feature_list
from src.utils.assets import AssetPaths, slugify, find_image
from src.utils.avatars import svg_avatar_data_uri
from src.utils.aliases import load_aliases

try:
    from src.analysis.metrics import compute_overall_metrics  # type: ignore
except Exception:
    compute_overall_metrics = None

from src.predict.whatif import build_feature_row  # type: ignore


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Predictive Serve",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎾",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


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
    "hero_headline": "Profesyonel tenis tahmin konsolu",
    "hero_kicker": "Gerçek veri boru hattı • Sportradar gelecek takvimi + oranlar • Logistic Regression + Elo/form",
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
    "hero_headline": "Professional tennis forecasting console",
    "hero_kicker": "End-to-end data pipeline • Sportradar upcoming + odds imagery • Classic ML baseline (LR + features)",
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
        padding-top: 1.25rem; 
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .ps-shell { margin-bottom: 0.75rem; }

    .ps-topbar {
        position: sticky;
        top: 0;
        z-index: 200;
        backdrop-filter: blur(14px);
        background: linear-gradient(to bottom, rgba(14, 17, 23, 0.92), rgba(14, 17, 23, 0.65));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    .ps-brand { font-weight: 850; letter-spacing: 0.02em; font-size: 1.05rem; color:#fff;}
    .ps-badge { opacity: 0.8; font-size: 0.85rem;}
    .ps-hero {
        margin-top: 12px;
        padding: 18px 18px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        background: radial-gradient(1200px 400px at 20% -10%, rgba(75,156,255,0.22), transparent 55%),
                    radial-gradient(800px 300px at 90% -20%, rgba(255,75,75,0.18), transparent 45%),
                    rgba(255,255,255,0.025);
    }
    .ps-hero-h1 {
        margin: 0;
        font-size: 2.0rem;
        line-height: 1.08;
        font-weight: 900;
        letter-spacing: -0.02em;
    }
    .ps-hero-kicker { margin-top: 8px; opacity: 0.75; font-size: 0.95rem;}
    .ps-avatar-frame {
      width: 100%;
      max-width: 180px;
      margin: 0 auto;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.03);
      aspect-ratio: 1 / 1;
      display:flex; align-items:center; justify-content:center;
    }
    .ps-avatar-frame img { width: 100%; height: auto; display:block; }

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


def render_site_header(t: Dict[str, str]) -> None:
    tit = html.escape(str(t.get("title", "Predictive Serve")))
    head = html.escape(str(t.get("hero_headline", "")))
    kick = html.escape(str(t.get("hero_kicker", "")))
    st.markdown(
        f"""
        <div class="ps-shell">
          <div class="ps-topbar">
            <div class="ps-brand">🎾 {tit}</div>
          </div>
          <div class="ps-hero">
            <h1 class="ps-hero-h1">{head}</h1>
            <div class="ps-hero-kicker">{kick}</div>
          </div>
        </div>
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

FIXTURES_CANDIDATES = [
    PROCESSED_DIR / "fixtures_upcoming.csv",
    PROCESSED_DIR / "upcoming_fixtures.csv",
    PROCESSED_DIR / "fixtures.csv",
    PROJECT_ROOT / "data" / "examples" / "fixtures_upcoming.csv",
]

ASSETS = AssetPaths(PROJECT_ROOT / "assets")


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


def normalize_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixtures (upcoming matches) schema normalization.
    Expected columns (best effort): date, tournament, surface, round, playerA, playerB, oddsA, oddsB, match_id
    """
    df = df.copy()

    # Rename common alternates
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
    if "player_1" in df.columns and "playerA" not in df.columns:
        df = df.rename(columns={"player_1": "playerA"})
    if "player_2" in df.columns and "playerB" not in df.columns:
        df = df.rename(columns={"player_2": "playerB"})
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["tournament", "surface", "round", "playerA", "playerB", "match_id"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)

    # Odds numeric
    for c in ["oddsA", "oddsB"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Minimal required columns check later in UI
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


@st.cache_data(show_spinner=False)
def load_fixtures() -> Tuple[pd.DataFrame, str]:
    path = first_existing(FIXTURES_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Fixtures dataset not found.")
    df = pd.read_csv(path)
    df = normalize_fixtures(df)
    return df, str(path)


@st.cache_data(show_spinner=False)
def score_fixtures(
    fixtures_df: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Batch-score fixtures using the existing single-row feature builder.
    Notes:
    - Keeps odds optional.
    - Uses fixture date as cutoff (no leakage).
    """
    required = ["date", "surface", "playerA", "playerB"]
    for c in required:
        if c not in fixtures_df.columns:
            raise ValueError(f"Fixtures missing required column: {c}")

    # Precompute last-seen date for each player in history for fast "has snapshot" checks.
    last_seen: Dict[str, pd.Timestamp] = {}
    if {"date", "playerA", "playerB"}.issubset(set(history_df.columns)):
        hd = history_df[["date", "playerA", "playerB"]].copy()
        hd["date"] = pd.to_datetime(hd["date"], errors="coerce")
        hd = hd.dropna(subset=["date"])
        for col in ["playerA", "playerB"]:
            tmp = hd[[col, "date"]].dropna()
            if not tmp.empty:
                grp = tmp.groupby(col, as_index=True)["date"].max()
                for k, v in grp.items():
                    if isinstance(k, str) and pd.notna(v):
                        last_seen[k] = pd.Timestamp(v)

    rows: List[Dict[str, Any]] = []
    for _, r in fixtures_df.iterrows():
        date = r.get("date")
        surface = r.get("surface")
        playerA = r.get("playerA")
        playerB = r.get("playerB")

        if pd.isna(date) or not surface or not playerA or not playerB:
            continue

        round_code = r.get("round") if "round" in fixtures_df.columns else None
        oddsA = float(r["oddsA"]) if ("oddsA" in fixtures_df.columns and pd.notna(r.get("oddsA"))) else None
        oddsB = float(r["oddsB"]) if ("oddsB" in fixtures_df.columns and pd.notna(r.get("oddsB"))) else None

        # Snapshot availability (best effort): we have history if last match date is before fixture date.
        snapA_ok = False
        snapB_ok = False
        try:
            dts = pd.Timestamp(date)
            snapA_ok = (playerA in last_seen) and (last_seen[playerA] < dts)
            snapB_ok = (playerB in last_seen) and (last_seen[playerB] < dts)
        except Exception:
            pass

        row_df = build_feature_row(
            history=history_df,
            feature_cols=feature_cols,
            playerA=playerA,
            playerB=playerB,
            surface=str(surface),
            date=pd.Timestamp(date),
            round_code=str(round_code) if (round_code and str(round_code).strip()) else None,
            oddsA=oddsA,
            oddsB=oddsB,
        )
        rows.append(
            {
                "match_id": r.get("match_id", ""),
                "date": pd.Timestamp(date),
                "tournament": r.get("tournament", ""),
                "round": r.get("round", ""),
                "surface": str(surface),
                "playerA": str(playerA),
                "playerB": str(playerB),
                "oddsA": oddsA,
                "oddsB": oddsB,
                "snapA_ok": int(snapA_ok),
                "snapB_ok": int(snapB_ok),
                "feature_row": row_df,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Predict
    model, imputer, _ = load_artifacts()
    X = pd.concat(out["feature_row"].tolist(), ignore_index=True)
    X_imp = imputer.transform(X[feature_cols])
    out["p_model"] = model.predict_proba(X_imp)[:, 1].astype(float)
    out = out.drop(columns=["feature_row"])

    # Market implied prob and edge when odds exist
    if out["oddsA"].notna().any() and out["oddsB"].notna().any():
        invA = 1.0 / out["oddsA"].astype(float)
        invB = 1.0 / out["oddsB"].astype(float)
        denom = invA + invB
        out["pA_market"] = invA / denom
        out["edge"] = out["p_model"] - out["pA_market"]
    else:
        out["pA_market"] = np.nan
        out["edge"] = np.nan

    return out


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

def _player_elo_series(history_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Long-form Elo series for a single player."""
    df = history_df.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "elo", "elo_surface"])
    df = df[df["date"].notna()].copy()
    maskA = df.get("playerA", pd.Series("", index=df.index)).astype(str) == player
    maskB = df.get("playerB", pd.Series("", index=df.index)).astype(str) == player
    a = df.loc[maskA, ["date"]].copy()
    b = df.loc[maskB, ["date"]].copy()
    a["elo"] = pd.to_numeric(df.loc[maskA].get("eloA", np.nan), errors="coerce")
    b["elo"] = pd.to_numeric(df.loc[maskB].get("eloB", np.nan), errors="coerce")
    a["elo_surface"] = pd.to_numeric(df.loc[maskA].get("elo_surfaceA", np.nan), errors="coerce")
    b["elo_surface"] = pd.to_numeric(df.loc[maskB].get("elo_surfaceB", np.nan), errors="coerce")
    out = pd.concat([a, b], ignore_index=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _player_recent_matches(history_df: pd.DataFrame, player: str, limit: int = 50) -> pd.DataFrame:
    df = history_df.copy()
    if "date" in df.columns:
        df = df[df["date"].notna()].copy()
    a0 = df.get("playerA", pd.Series("", index=df.index)).astype(str)
    b0 = df.get("playerB", pd.Series("", index=df.index)).astype(str)
    df = df[(a0 == player) | (b0 == player)].copy()
    if df.empty:
        return df
    df = df.sort_values("date", ascending=False)
    # Recompute A/B on the filtered frame to avoid length mismatch
    a = df.get("playerA", pd.Series("", index=df.index)).astype(str)
    b = df.get("playerB", pd.Series("", index=df.index)).astype(str)
    df["opponent"] = np.where(a == player, b, a)
    # Determine win best-effort
    if "winner" in df.columns:
        df["result"] = np.where(df["winner"].astype(str) == player, "W", "L")
    else:
        # In matches_with_* pipeline, playerA is winner at that stage
        df["result"] = np.where(a == player, "W", "L")
    cols = [c for c in ["date", "tournament", "round", "surface", "opponent", "result"] if c in df.columns]
    return df[cols].head(limit).copy()


def render_player_profile(player: str, history_df: pd.DataFrame, pred_df: pd.DataFrame):
    st.markdown("<div class='ps-card'><div class='ps-title'>👤 Player Profile</div></div>", unsafe_allow_html=True)

    img = find_image(ASSETS.players / slugify(player))
    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
    with c1:
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.markdown(
                f"<div class='ps-avatar-frame'><img src='{svg_avatar_data_uri(player)}' alt=''/></div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(f"### {player}")
        st.caption("Elo, form, and match history based on the processed dataset.")

    series = _player_elo_series(history_df, player)
    recent = _player_recent_matches(history_df, player, limit=50)
    # Full matchset for deeper stats
    full = history_df.copy()
    if "date" in full.columns:
        full["date"] = pd.to_datetime(full["date"], errors="coerce")
        full = full[full["date"].notna()].copy()
    a = full.get("playerA", pd.Series("", index=full.index)).astype(str)
    b = full.get("playerB", pd.Series("", index=full.index)).astype(str)
    full = full[(a == player) | (b == player)].copy()
    if not full.empty:
        # Recompute A/B on the filtered frame to avoid length mismatch
        a2 = full.get("playerA", pd.Series("", index=full.index)).astype(str)
        b2 = full.get("playerB", pd.Series("", index=full.index)).astype(str)
        full["opponent"] = np.where(a2 == player, b2, a2)
        if "winner" in full.columns:
            full["result"] = np.where(full["winner"].astype(str) == player, "W", "L")
        else:
            full["result"] = np.where(a2 == player, "W", "L")

    # KPIs
    matches = int(len(recent)) if not recent.empty else 0
    wins = int((recent.get("result") == "W").sum()) if ("result" in recent.columns) else 0
    win_rate = (wins / matches * 100.0) if matches > 0 else np.nan
    elo_last = float(series["elo"].dropna().iloc[-1]) if (not series.empty and series["elo"].notna().any()) else np.nan
    elo_s_last = float(series["elo_surface"].dropna().iloc[-1]) if (not series.empty and series["elo_surface"].notna().any()) else np.nan

    with c3:
        st.metric("Matches", f"{matches:,}")
        st.metric("Win rate", "—" if pd.isna(win_rate) else f"{win_rate:.1f}%")
    with c4:
        st.metric("Latest Elo", "—" if pd.isna(elo_last) else f"{elo_last:.0f}")
        st.metric("Latest Surface Elo", "—" if pd.isna(elo_s_last) else f"{elo_s_last:.0f}")

    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>📈 Elo (history)</div>", unsafe_allow_html=True)
    if series.empty:
        st.caption("No Elo history found for this player.")
    else:
        plot_df = series.dropna(subset=["elo"]).copy()
        chart = (
            alt.Chart(plot_df)
            .mark_line(opacity=0.9, strokeWidth=3)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("elo:Q", title="Elo", scale=alt.Scale(zero=False)),
                tooltip=["date:T", "elo:Q"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Performance breakdowns
    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>📊 Performance breakdown</div>", unsafe_allow_html=True)
    if full.empty:
        st.caption("No matches available for breakdown.")
    else:
        # Surface win rates
        if "surface" in full.columns:
            surf = full[["surface", "result"]].dropna().copy()
            if not surf.empty:
                agg = (
                    surf.assign(win=(surf["result"] == "W").astype(int), match=1)
                    .groupby("surface", as_index=False)
                    .agg(matches=("match", "sum"), wins=("win", "sum"))
                )
                agg["win_rate"] = (agg["wins"] / agg["matches"]) * 100.0
                agg = agg.sort_values(["matches", "win_rate"], ascending=[False, False])
                cbs1, cbs2 = st.columns([1, 1])
                with cbs1:
                    st.markdown("**By surface**")
                    st.dataframe(
                        agg.rename(columns={"surface": "Surface", "matches": "Matches", "wins": "Wins", "win_rate": "Win Rate (%)"})
                        .round({"Win Rate (%)": 1}),
                        use_container_width=True,
                        hide_index=True,
                        height=240,
                    )
                with cbs2:
                    st.markdown("**Surface win rate chart**")
                    st.altair_chart(
                        alt.Chart(agg).mark_bar().encode(
                            x=alt.X("win_rate:Q", title="Win Rate (%)"),
                            y=alt.Y("surface:N", sort="-x", title="Surface"),
                            tooltip=["surface", "matches", "wins", alt.Tooltip("win_rate:Q", format=".1f")],
                        ),
                        use_container_width=True,
                    )

        # Best tournaments (by win rate, min matches)
        if "tournament" in full.columns:
            tour = full[["tournament", "result"]].dropna().copy()
            if not tour.empty:
                agg2 = (
                    tour.assign(win=(tour["result"] == "W").astype(int), match=1)
                    .groupby("tournament", as_index=False)
                    .agg(matches=("match", "sum"), wins=("win", "sum"))
                )
                agg2["win_rate"] = (agg2["wins"] / agg2["matches"]) * 100.0
                agg2 = agg2[agg2["matches"] >= 5].sort_values(["win_rate", "matches"], ascending=[False, False]).head(15)
                st.markdown("**Best tournaments (min 5 matches)**")
                st.dataframe(
                    agg2.rename(columns={"tournament": "Tournament", "matches": "Matches", "wins": "Wins", "win_rate": "Win Rate (%)"})
                    .round({"Win Rate (%)": 1}),
                    use_container_width=True,
                    hide_index=True,
                    height=280,
                )
                # Quick navigation to tournament profile
                tourn_opts = ["—"] + agg2["tournament"].astype(str).tolist()
                pick_t = st.selectbox("Open a tournament profile", options=tourn_opts, index=0, key="player_best_tourn_pick")
                if pick_t != "—" and st.button("Open Tournament Profile", key="open_tournament_from_player"):
                    st.session_state["profile_tournament"] = str(pick_t)
                    st.info("Go to the **Tournaments** tab (preselected).")
    st.markdown("</div>", unsafe_allow_html=True)

    # H2H module
    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>🤝 Head-to-head (H2H)</div>", unsafe_allow_html=True)
    if full.empty:
        st.caption("No match history to compute H2H.")
    else:
        opps = sorted(set([clean_text(x) for x in full["opponent"].dropna().astype(str).tolist() if clean_text(x)]))
        if not opps:
            st.caption("No opponents found.")
        else:
            opp = st.selectbox("Select opponent", options=opps, index=0, key="h2h_opponent_select")

            h2h_df = full[full["opponent"].astype(str) == opp].copy()
            total_h2h = len(h2h_df)
            wins_h2h = int((h2h_df["result"] == "W").sum()) if "result" in h2h_df.columns else 0
            wr_h2h = (wins_h2h / total_h2h * 100.0) if total_h2h > 0 else np.nan

            # Latest snapshots for both players (best effort from last match involving each)
            def last_snapshot(p: str) -> Tuple[float, float, float]:
                s = _player_elo_series(history_df, p)
                elo = float(s["elo"].dropna().iloc[-1]) if (not s.empty and s["elo"].notna().any()) else np.nan
                es = float(s["elo_surface"].dropna().iloc[-1]) if (not s.empty and s["elo_surface"].notna().any()) else np.nan
                # approximate recent form from last 10 matches in full for that player
                pm = _player_recent_matches(history_df, p, limit=10)
                if not pm.empty and "result" in pm.columns:
                    f = float((pm["result"] == "W").mean() * 100.0)
                else:
                    f = np.nan
                return elo, es, f

            elo_p, eloS_p, form_p = last_snapshot(player)
            elo_o, eloS_o, form_o = last_snapshot(opp)

            c_h1, c_h2, c_h3, c_h4 = st.columns(4)
            with c_h1:
                st.metric("H2H matches", f"{total_h2h:,}")
            with c_h2:
                st.metric("H2H win rate", "—" if pd.isna(wr_h2h) else f"{wr_h2h:.1f}%")
            with c_h3:
                st.metric("Elo (you vs opp)", f"{elo_p:.0f} vs {elo_o:.0f}" if (pd.notna(elo_p) and pd.notna(elo_o)) else "—")
            with c_h4:
                st.metric("Form10 (you vs opp)", f"{form_p:.0f}% vs {form_o:.0f}%" if (pd.notna(form_p) and pd.notna(form_o)) else "—")

            cols = [c for c in ["date", "tournament", "round", "surface", "result"] if c in h2h_df.columns]
            if "date" in h2h_df.columns:
                h2h_df = h2h_df.sort_values("date", ascending=False)
            st.dataframe(h2h_df[cols].head(30), use_container_width=True, hide_index=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    # Last 12 months form (rolling)
    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>🧭 Form (last 12 months)</div>", unsafe_allow_html=True)
    if full.empty or "date" not in full.columns:
        st.caption("No data for form trend.")
    else:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
        f12 = full[full["date"] >= cutoff].sort_values("date").copy()
        if f12.empty:
            st.caption("No matches in last 12 months.")
        else:
            f12["win"] = (f12["result"] == "W").astype(int)
            f12["rolling_wr_10"] = f12["win"].rolling(10, min_periods=3).mean() * 100.0
            chart = alt.Chart(f12).mark_line(opacity=0.9, strokeWidth=3).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("rolling_wr_10:Q", title="Rolling win rate (last 10 matches, %)"),
                tooltip=["date:T", "result", "opponent", alt.Tooltip("rolling_wr_10:Q", format=".1f")],
            )
            st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>🗓️ Recent matches</div>", unsafe_allow_html=True)
    if recent.empty:
        st.caption("No matches found.")
    else:
        st.dataframe(recent, use_container_width=True, hide_index=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)


def render_tournament_profile(tournament: str, history_df: pd.DataFrame, pred_df: pd.DataFrame):
    st.markdown("<div class='ps-card'><div class='ps-title'>🏆 Tournament Profile</div></div>", unsafe_allow_html=True)

    img = find_image(ASSETS.tournaments / slugify(tournament))
    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
    with c1:
        if img:
            st.image(str(img), use_container_width=True)
    with c2:
        st.markdown(f"### {tournament}")
        st.caption("Tournament match history and surface/round distribution.")

    df = history_df.copy()
    if "tournament" not in df.columns:
        st.warning("Tournament column not present in history dataset.")
        return
    df = df[df["tournament"].astype(str) == tournament].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].copy()

    total = len(df)
    first_date = df["date"].min().date() if (total > 0 and "date" in df.columns) else None
    last_date = df["date"].max().date() if (total > 0 and "date" in df.columns) else None
    surface_mode = df["surface"].mode().iloc[0] if (total > 0 and "surface" in df.columns and df["surface"].notna().any()) else "—"

    with c3:
        st.metric("Matches", f"{total:,}")
        st.metric("Surface (mode)", str(surface_mode))
    with c4:
        st.metric("First date", "—" if first_date is None else str(first_date))
        st.metric("Last date", "—" if last_date is None else str(last_date))

    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>📊 Distribution</div>", unsafe_allow_html=True)
    cdist1, cdist2 = st.columns(2)
    with cdist1:
        if total > 0 and "surface" in df.columns:
            s_cnt = df["surface"].value_counts().reset_index()
            s_cnt.columns = ["surface", "count"]
            st.altair_chart(
                alt.Chart(s_cnt).mark_bar().encode(x="count:Q", y=alt.Y("surface:N", sort="-x"), tooltip=["surface", "count"]),
                use_container_width=True,
            )
    with cdist2:
        if total > 0 and "round" in df.columns:
            r_cnt = df["round"].fillna("—").astype(str).value_counts().head(15).reset_index()
            r_cnt.columns = ["round", "count"]
            st.altair_chart(
                alt.Chart(r_cnt).mark_bar().encode(x="count:Q", y=alt.Y("round:N", sort="-x"), tooltip=["round", "count"]),
                use_container_width=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>🗓️ Recent matches</div>", unsafe_allow_html=True)
    if total == 0:
        st.caption("No matches found.")
    else:
        cols = [c for c in ["date", "round", "surface", "playerA", "playerB"] if c in df.columns]
        st.dataframe(df.sort_values("date", ascending=False)[cols].head(100), use_container_width=True, hide_index=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    # Top players in this tournament
    st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ps-title'>⭐ Top players (this tournament)</div>", unsafe_allow_html=True)
    if total == 0 or not {"playerA", "playerB"}.issubset(set(df.columns)):
        st.caption("Not enough data.")
    else:
        # Approx wins: in matches_with_* data playerA is winner
        winners = df["playerA"].astype(str)
        counts = winners.value_counts().head(25).reset_index()
        counts.columns = ["player", "wins"]
        st.dataframe(counts, use_container_width=True, hide_index=True, height=360)

        pick = st.selectbox("Open a player profile from this list", options=["—"] + counts["player"].tolist(), index=0, key="tourn_top_players_pick")
        if pick != "—" and st.button("Open Player Profile", key="open_player_from_tournament"):
            st.session_state["profile_player"] = str(pick)
            st.info("Go to the **Players** tab (preselected).")
    st.markdown("</div>", unsafe_allow_html=True)

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
render_site_header(t)
render_kpis(filtered_for_kpi, t)

if dev_mode:
    st.caption(f"[DEV] predictions source: {pred_path} | rows(all)={len(pred_df):,} rows(filtered)={len(filtered_for_kpi):,}")
    st.caption(f"[DEV] columns: {', '.join(list(pred_df.columns)[:30])}{' ...' if len(pred_df.columns)>30 else ''}")

st.divider()

tab_matches, tab_upcoming, tab_players, tab_tournaments, tab_whatif, tab_leaderboard = st.tabs(
    [t["tabs_matches"], "Upcoming", "Players", "Tournaments", t["tabs_whatif"], t["tabs_leaderboard"]]
)


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

        # Quick links to profiles
        prof_c1, prof_c2, prof_c3 = st.columns([1, 1, 2])
        with prof_c1:
            if st.button("Open Player A Profile", key="open_playerA_from_match"):
                st.session_state["profile_player"] = str(simulated_row.get("playerA", ""))
                st.info("Go to the **Players** tab (preselected).")
        with prof_c2:
            if st.button("Open Player B Profile", key="open_playerB_from_match"):
                st.session_state["profile_player"] = str(simulated_row.get("playerB", ""))
                st.info("Go to the **Players** tab (preselected).")
        with prof_c3:
            if st.button("Open Tournament Profile", key="open_tournament_from_match"):
                st.session_state["profile_tournament"] = str(simulated_row.get("tournament", ""))
                st.info("Go to the **Tournaments** tab (preselected).")

        render_match_card(simulated_row, t)

    if dev_mode:
        st.caption(f"[DEV] match filters => player={player_pick} tourn={tournaments} only_market={only_market} only_model={only_model} min_edge={min_edge}")


# =========================
# Upcoming tab
# =========================
with tab_upcoming:
    st.markdown("<div class='ps-card'><div class='ps-title'>Upcoming Matches</div></div>", unsafe_allow_html=True)

    # Load artifacts + history once here (cache handles speed)
    history_df, _history_path = load_history()
    model, imputer, feature_cols = load_artifacts()

    try:
        fixtures_df, fixtures_path = load_fixtures()
    except Exception as e:
        st.info("No fixtures file found yet. Create `data/processed/fixtures_upcoming.csv` (or use the example in `data/examples/`).")
        st.caption(f"Details: {e}")
        st.stop()

    # Basic validation
    required = ["date", "surface", "playerA", "playerB"]
    missing = [c for c in required if c not in fixtures_df.columns]
    if missing:
        st.error(f"Fixtures file missing columns: {', '.join(missing)}")
        st.caption(f"Source: {fixtures_path}")
        st.stop()

    # Default date filter: from today to +14 days when possible
    today = pd.Timestamp.today().normalize()
    dmin = fixtures_df["date"].min()
    dmax = fixtures_df["date"].max()
    if pd.notna(dmin) and pd.notna(dmax):
        default_start = max(today, dmin.normalize())
        default_end = min(dmax.normalize(), today + pd.Timedelta(days=14))
        if default_end < default_start:
            default_end = default_start
        dr2 = st.date_input("Date range", value=(default_start.date(), default_end.date()), key="u_dates")
        u_d1, u_d2, _ = parse_date_range(dr2, t)
        if u_d1 and u_d2:
            fixtures_df = fixtures_df[
                (fixtures_df["date"].dt.date >= u_d1) & (fixtures_df["date"].dt.date <= u_d2)
            ].copy()

    # Extra filters
    colf1, colf2, colf3 = st.columns([2, 1, 1])
    with colf1:
        tourn_opts2 = safe_unique(fixtures_df, "tournament")
        tourn_pick = st.multiselect("Tournament", tourn_opts2, default=[], key="u_tourn")
    with colf2:
        surf_opts2 = safe_unique(fixtures_df, "surface")
        surf_pick = st.multiselect("Surface", surf_opts2, default=surf_opts2, key="u_surf")
    with colf3:
        min_conf = st.slider("Min confidence", 0.50, 0.90, 0.55, 0.01, key="u_minconf")

    if tourn_pick and "tournament" in fixtures_df.columns:
        fixtures_df = fixtures_df[fixtures_df["tournament"].isin(tourn_pick)].copy()
    if surf_pick and "surface" in fixtures_df.columns:
        fixtures_df = fixtures_df[fixtures_df["surface"].isin(surf_pick)].copy()

    if fixtures_df.empty:
        st.warning("No fixtures after filters.")
        st.caption(f"Source: {fixtures_path}")
        st.stop()

    # Score fixtures
    # Apply aliases (canonical names) for scoring consistency
    aliases = load_aliases()
    for col in ["playerA", "playerB"]:
        if col in fixtures_df.columns:
            fixtures_df[col] = fixtures_df[col].astype(str).map(aliases.map_player)
    if "tournament" in fixtures_df.columns:
        fixtures_df["tournament"] = fixtures_df["tournament"].astype(str).map(aliases.map_tournament)

    scored = score_fixtures(fixtures_df, history_df, feature_cols)
    if scored.empty:
        st.warning("No fixtures could be scored (missing date/surface/player names).")
        st.caption(f"Source: {fixtures_path}")
        st.stop()

    # Confidence and winner label
    scored = scored.copy()
    scored["winner_pick"] = np.where(scored["p_model"] >= 0.5, scored["playerA"], scored["playerB"])
    scored["winner_prob"] = np.where(scored["p_model"] >= 0.5, scored["p_model"], 1.0 - scored["p_model"])
    scored = scored[scored["winner_prob"] >= float(min_conf)]
    scored = scored.sort_values(["date", "winner_prob"], ascending=[True, False])

    st.write(f"Matches found: **{len(scored):,}**")

    # Data quality report
    if "snapA_ok" in scored.columns and "snapB_ok" in scored.columns:
        missA = int((scored["snapA_ok"] == 0).sum())
        missB = int((scored["snapB_ok"] == 0).sum())
        if missA or missB:
            st.warning(
                f"Some fixtures have limited history snapshots: "
                f"missing A-snapshot={missA}, missing B-snapshot={missB}. "
                f"Add mappings in `data/registry/player_aliases.csv` if names differ."
            )

    def _initials(name: str) -> str:
        parts = [p for p in str(name).split() if p]
        if not parts:
            return "?"
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def _player_chip(name: str) -> str:
        img = find_image(ASSETS.players / slugify(name))
        if img:
            # local file rendering via Streamlit is easiest in st.image; for HTML, we show initials badge
            return f"<span style='display:inline-flex;align-items:center;gap:8px;'><span style='width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,0.12);display:inline-flex;align-items:center;justify-content:center;font-size:11px;'>{_initials(name)}</span><span>{name}</span></span>"
        return f"<span style='display:inline-flex;align-items:center;gap:8px;'><span style='width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,0.12);display:inline-flex;align-items:center;justify-content:center;font-size:11px;'>{_initials(name)}</span><span>{name}</span></span>"

    # Compact, less cluttered list as cards (shows avatars via initials; selected row shows real images below)
    st.markdown("<div class='ps-card'><div class='ps-title'>📅 Upcoming list</div></div>", unsafe_allow_html=True)
    preview = scored.head(40).copy()
    for i, r in preview.iterrows():
        date_s = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        tour = clean_text(r.get("tournament", ""))
        surface = clean_text(r.get("surface", ""))
        rnd = clean_text(r.get("round", ""))
        pa = clean_text(r.get("playerA", ""))
        pb = clean_text(r.get("playerB", ""))
        pick = clean_text(r.get("winner_pick", ""))
        conf = float(r.get("winner_prob", np.nan)) * 100.0
        st.markdown(
            f"""
            <div class="ps-card" style="padding:14px;">
              <div style="display:flex;justify-content:space-between;opacity:0.7;font-size:0.9rem;">
                <div>📅 {date_s}</div>
                <div>🏆 {tour} · 🎾 {surface} {rnd}</div>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                <div style="font-size:1.1rem;font-weight:700;">{_player_chip(pa)} <span style="opacity:0.5;font-weight:600;">vs</span> {_player_chip(pb)}</div>
                <div style="text-align:right;">
                  <div style="font-weight:800;">🤖 {pick}</div>
                  <div style="opacity:0.75;">{conf:.1f}%</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Also keep a table for power users (selection works)
    base_cols = ["date", "tournament", "round", "surface", "playerA", "playerB", "winner_pick", "winner_prob", "pA_market", "edge"]
    for ox in ["oddsA", "oddsB"]:
        if ox in scored.columns and ox not in base_cols:
            base_cols.append(ox)

    show = scored[[c for c in base_cols if c in scored.columns]].copy()
    show["winner_prob"] = (show["winner_prob"] * 100).round(1)
    if "pA_market" in show.columns:
        show["pA_market"] = (show["pA_market"] * 100).round(1)
    if "edge" in show.columns:
        show["edge"] = (show["edge"] * 100).round(1)

    st.markdown("<div class='ps-card'><div class='ps-title'>🔎 Table view</div></div>", unsafe_allow_html=True)
    st.dataframe(
        show.rename(
            columns={
                "date": "Date",
                "tournament": "Tournament",
                "round": "Round",
                "surface": "Surface",
                "playerA": "Player A",
                "playerB": "Player B",
                "winner_pick": "AI Pick",
                "winner_prob": "AI Confidence",
                "pA_market": "Market (A)",
                "edge": "Edge (A)",
                "oddsA": "Odds A (dec.)",
                "oddsB": "Odds B (dec.)",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=360,
        selection_mode="single-row",
        on_select="rerun",
        key="upcoming_selection",
    )

    selected_rows_u = st.session_state.upcoming_selection.get("selection", {}).get("rows", [])
    st.divider()

    if selected_rows_u:
        idx_loc = selected_rows_u[0]
        original_idx = show.index[idx_loc]
        row = scored.loc[original_idx]

        # Quick links to profiles
        uprof_c1, uprof_c2, uprof_c3 = st.columns([1, 1, 2])
        with uprof_c1:
            if st.button("Open Player A Profile", key="open_playerA_from_upcoming"):
                st.session_state["profile_player"] = str(row.get("playerA", ""))
                st.info("Go to the **Players** tab (preselected).")
        with uprof_c2:
            if st.button("Open Player B Profile", key="open_playerB_from_upcoming"):
                st.session_state["profile_player"] = str(row.get("playerB", ""))
                st.info("Go to the **Players** tab (preselected).")
        with uprof_c3:
            if st.button("Open Tournament Profile", key="open_tournament_from_upcoming"):
                st.session_state["profile_tournament"] = str(row.get("tournament", ""))
                st.info("Go to the **Tournaments** tab (preselected).")

        # Images (optional)
        pa_slug = slugify(row["playerA"])
        pb_slug = slugify(row["playerB"])
        ta_slug = slugify(row.get("tournament", ""))

        imgA = find_image(ASSETS.players / pa_slug)
        imgB = find_image(ASSETS.players / pb_slug)
        imgT = find_image(ASSETS.tournaments / ta_slug)

        pa_name = str(row.get("playerA", "") or "")
        pb_name = str(row.get("playerB", "") or "")

        cimg1, cimg2, cimg3 = st.columns([1, 1, 2])
        with cimg1:
            if imgA:
                st.image(str(imgA), caption=row["playerA"], use_container_width=True)
            else:
                st.markdown(
                    f"<div class='ps-avatar-frame'><img src='{svg_avatar_data_uri(pa_name)}' alt=''></div>",
                    unsafe_allow_html=True,
                )
                st.caption(pa_name)
        with cimg2:
            if imgB:
                st.image(str(imgB), caption=row["playerB"], use_container_width=True)
            else:
                st.markdown(
                    f"<div class='ps-avatar-frame'><img src='{svg_avatar_data_uri(pb_name)}' alt=''></div>",
                    unsafe_allow_html=True,
                )
                st.caption(pb_name)
        with cimg3:
            tn = str(row.get("tournament", "") or "")
            if imgT:
                st.image(str(imgT), caption=tn, use_container_width=True)
            elif tn.strip():
                st.markdown(
                    f"<div class='ps-avatar-frame'><img src='{svg_avatar_data_uri(tn)}' alt=''></div>",
                    unsafe_allow_html=True,
                )
                st.caption(tn)
            else:
                st.caption("—")

        # Simple card
        oa = row.get("oddsA")
        ob = row.get("oddsB")
        odds_line = ""
        if pd.notna(oa) and pd.notna(ob):
            odds_line = f"<div style='margin-top:8px; opacity:0.85;'>📉 Decimal odds — A <b>{float(oa):.3f}</b> · B <b>{float(ob):.3f}</b></div>"

        st.markdown(
            f"""
            <div class="ps-card">
              <div style="opacity:0.7;">📅 {pd.to_datetime(row['date']).strftime('%Y-%m-%d')} | 🏆 {row.get('tournament','')} | 🎾 {row.get('surface','')} {row.get('round','')}</div>
              <div style="font-size:1.6rem; font-weight:800; margin-top:10px;">{row['playerA']} vs {row['playerB']}</div>
              <div style="margin-top:10px; font-size:1.1rem;">🤖 Pick: <b>{row['winner_pick']}</b> with <b>{row['winner_prob']*100:.1f}%</b></div>
              {odds_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if dev_mode:
        st.caption(f"[DEV] fixtures source: {fixtures_path} | rows={len(fixtures_df):,}")


# =========================
# Players tab
# =========================
with tab_players:
    st.markdown("<div class='ps-card'><div class='ps-title'>Players</div></div>", unsafe_allow_html=True)
    history_df, _ = load_history()

    # canonical mapping
    aliases = load_aliases()
    all_players = sorted(
        pd.unique(
            pd.concat(
                [history_df.get("playerA", pd.Series(dtype=str)), history_df.get("playerB", pd.Series(dtype=str))],
                ignore_index=True,
            )
        ).astype(str)
    )
    # Map only raw variants to canonical display if provided; otherwise keep original casing.
    all_players = [aliases.map_player(clean_text(p)) for p in all_players if clean_text(p)]
    all_players = sorted(set([p for p in all_players if p]))

    default_player = st.session_state.get("profile_player") if "profile_player" in st.session_state else None
    if default_player and default_player in all_players:
        default_idx = all_players.index(default_player)
    else:
        default_idx = 0

    if not all_players:
        st.info("No players found in history dataset.")
        st.stop()

    player_sel = st.selectbox("Select player", options=all_players, index=default_idx, key="player_profile_select")
    st.session_state["profile_player"] = player_sel
    render_player_profile(player_sel, history_df, pred_df)

    # Optional: show upcoming fixtures for this player
    try:
        fx, _ = load_fixtures()
        if {"playerA", "playerB"}.issubset(set(fx.columns)):
            fx = fx.copy()
            fx["playerA"] = fx["playerA"].astype(str).map(aliases.map_player)
            fx["playerB"] = fx["playerB"].astype(str).map(aliases.map_player)
            m = (fx["playerA"] == player_sel) | (fx["playerB"] == player_sel)
            fxp = fx[m].copy()
            if not fxp.empty:
                st.markdown("<div class='ps-card'>", unsafe_allow_html=True)
                st.markdown("<div class='ps-title'>⏳ Upcoming fixtures</div>", unsafe_allow_html=True)
                cols = [c for c in ["date", "tournament", "surface", "round", "playerA", "playerB", "oddsA", "oddsB"] if c in fxp.columns]
                st.dataframe(fxp.sort_values("date")[cols].head(50), use_container_width=True, hide_index=True, height=320)
                st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass


# =========================
# Tournaments tab
# =========================
with tab_tournaments:
    st.markdown("<div class='ps-card'><div class='ps-title'>Tournaments</div></div>", unsafe_allow_html=True)
    history_df, _ = load_history()
    aliases = load_aliases()
    if "tournament" not in history_df.columns:
        st.info("Tournament column not available in history dataset.")
        st.stop()

    tournaments_all = sorted(
        set([aliases.map_tournament(clean_text(x)) for x in history_df["tournament"].dropna().astype(str).tolist() if clean_text(x)])
    )
    if not tournaments_all:
        st.info("No tournaments found in history dataset.")
        st.stop()

    default_t = st.session_state.get("profile_tournament") if "profile_tournament" in st.session_state else None
    default_idx_t = tournaments_all.index(default_t) if (default_t in tournaments_all) else 0
    tourn_sel = st.selectbox("Select tournament", options=tournaments_all, index=default_idx_t, key="tourn_profile_select")
    st.session_state["profile_tournament"] = tourn_sel
    render_tournament_profile(tourn_sel, history_df, pred_df)

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
