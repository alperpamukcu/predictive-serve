"""
Predictive Serve — professional tennis forecasting console.

English-only single-page app with six views:
    1. Matches    - historical predictions explorer (with right/wrong colour cues)
    2. Upcoming   - model-scored fixtures (API-Tennis, with demo fallback)
    3. Players    - per-player profile, with player photo (API or local cache)
    4. Tournaments- per-tournament profile
    5. What-if    - single-match predictor
    6. Leaderboard- aggregate player rankings

Navigation is button-driven so the app can deep-link from any cell to a
player or tournament profile in a single click.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import random
import re
import subprocess
import sys
from html import escape as h
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from joblib import load as joblib_load

from src.utils.assets import find_image, slugify
from src.utils.avatars import svg_avatar_data_uri
from src.utils.config import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import getenv, try_load_dotenv
from src.utils.feature_utils import load_feature_list

try:
    from src.predict.whatif import build_feature_row  # type: ignore
except Exception:  # pragma: no cover
    build_feature_row = None  # type: ignore

try:
    from src.integrations.api_tennis import ApiTennisConfig, get_fixtures  # type: ignore
except Exception:  # pragma: no cover
    ApiTennisConfig = None  # type: ignore
    get_fixtures = None  # type: ignore


# =============================================================================
# Page + assets
# =============================================================================

st.set_page_config(
    page_title="Predictive Serve",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


CSS = """
<style>
header[data-testid="stHeader"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
#MainMenu, footer { visibility: hidden; }

:root {
  --bg: #131a26;
  --bg-2: #1a2334;
  --surface: rgba(255,255,255,0.05);
  --surface-2: rgba(255,255,255,0.085);
  --surface-strong: rgba(255,255,255,0.12);
  --line: rgba(255,255,255,0.12);
  --line-strong: rgba(255,255,255,0.22);
  --text: #f1f4fa;
  --muted: rgba(241,244,250,0.78);
  --soft-muted: rgba(241,244,250,0.55);
  --accent: #6aa9ff;
  --accent-2: #9c87ff;
  --warm: #ff8d63;
  --good: #2dd29a;
  --good-soft: rgba(45, 210, 154, 0.16);
  --bad: #ff6471;
  --bad-soft: rgba(255, 100, 113, 0.16);
  --radius: 14px;
  --radius-lg: 20px;
}

.stApp {
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(106,169,255,0.10), transparent 55%),
    radial-gradient(900px 520px at 92% -10%, rgba(156,135,255,0.08), transparent 55%),
    radial-gradient(800px 600px at 50% 110%, rgba(255,141,99,0.04), transparent 50%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
}

/* Centre the column and give it a sane max-width */
.block-container {
  padding-top: 0.6rem !important;
  padding-left: 1.6rem !important;
  padding-right: 1.6rem !important;
  padding-bottom: 3rem;
  max-width: 1280px;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Top nav */
.ps-nav {
  position: sticky; top: 0; z-index: 240;
  margin: 0 -1.6rem 1rem -1.6rem;
  padding: 14px 24px;
  backdrop-filter: blur(18px) saturate(140%);
  background: linear-gradient(180deg, rgba(7,9,13,0.92), rgba(7,9,13,0.66));
  border-bottom: 1px solid var(--line);
  display: flex; justify-content: space-between; align-items: center; gap: 14px;
}
.ps-nav-left { display:flex; align-items:center; gap:14px; }
.ps-logo {
  width:34px; height:34px; border-radius:10px;
  display:inline-flex; align-items:center; justify-content:center;
  background: conic-gradient(from 220deg, #4ea1ff, #7c5cff, #ff7a3d, #4ea1ff);
  color:#0b0d12; font-weight:900; font-size:0.95rem;
  box-shadow: 0 8px 24px rgba(78,161,255,0.25), inset 0 1px 0 rgba(255,255,255,0.4);
}
.ps-brand { font-weight:800; letter-spacing:-0.01em; font-size:1.05rem; color:#fff; }
.ps-brand small {
  display:block; font-weight:500; font-size:0.72rem;
  color:var(--muted); margin-top:-2px;
  letter-spacing:0.04em; text-transform:uppercase;
}
.ps-pill {
  display:inline-flex; align-items:center; gap:6px;
  padding:5px 11px; border-radius:999px;
  background:var(--surface); border:1px solid var(--line);
  color:var(--text); font-size:0.78rem; font-weight:600;
}
.ps-pill .dot { width:7px; height:7px; border-radius:50%; background:var(--good); }

/* Hero */
.ps-hero {
  margin: 4px 0 18px 0;
  padding: 26px 28px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--line);
  background:
    radial-gradient(800px 240px at 8% -20%, rgba(78,161,255,0.18), transparent 55%),
    radial-gradient(600px 220px at 92% -10%, rgba(255,122,61,0.14), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
}
.ps-eyebrow {
  display:inline-flex; gap:6px; padding:5px 11px; border-radius:999px;
  background: rgba(78,161,255,0.10); border:1px solid rgba(78,161,255,0.30);
  color:#b8d6ff; font-size:0.74rem; font-weight:700;
  letter-spacing:0.06em; text-transform:uppercase;
}
.ps-hero h1 {
  margin: 14px 0 6px 0;
  font-size: clamp(1.7rem, 3vw, 2.4rem);
  line-height:1.06; font-weight:800; letter-spacing:-0.02em; color:#fff;
}
.ps-hero h1 span.grad {
  background: linear-gradient(90deg, #ffffff 0%, #b8d6ff 50%, #ffd0bf 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.ps-hero p { margin:0; color:var(--muted); font-size:1.0rem; max-width:820px; }

/* KPI grid */
.ps-kpi-grid {
  display:grid; grid-template-columns: repeat(4, minmax(0,1fr));
  gap:12px; margin: 10px 0 18px 0;
}
@media (max-width: 1100px) { .ps-kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.ps-kpi {
  padding:14px 16px; border-radius: var(--radius);
  border:1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  position:relative; overflow:hidden;
}
.ps-kpi::after {
  content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); opacity:.55;
}
.ps-kpi-label { font-size:0.72rem; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.ps-kpi-val { font-size:1.55rem; font-weight:800; color:#fff; margin-top:6px; letter-spacing:-0.01em; }
.ps-kpi-sub { font-size:0.78rem; color:var(--muted); margin-top:4px; }

/* Cards */
.ps-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  border:1px solid var(--line); border-radius: var(--radius);
  padding:18px; margin-bottom:14px;
}
.ps-card.tight { padding:12px 14px; }
.ps-section-title {
  font-size:1.05rem; font-weight:700; color:#fff; letter-spacing:-0.01em; margin:10px 0 12px 0;
}

/* Button-based nav (replaces st.tabs to fix layout AND enable programmatic navigation) */
.ps-tabs { margin-bottom: 14px; }
.ps-tabs > div[data-testid="stHorizontalBlock"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 4px;
  gap: 4px !important;
  width: fit-content !important;
}
.ps-tabs .stButton > button {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: 9px !important;
  padding: 8px 16px !important;
  font-weight: 600 !important;
  border: 1px solid transparent !important;
  font-size: 0.92rem !important;
  white-space: nowrap !important;
}
.ps-tabs .stButton > button:hover {
  color: #fff !important;
  background: var(--surface-2) !important;
}
.ps-tabs .stButton > button[kind="primary"] {
  background: linear-gradient(180deg, rgba(78,161,255,0.22), rgba(78,161,255,0.08)) !important;
  color: #fff !important;
  border-color: rgba(78,161,255,0.40) !important;
  box-shadow: none !important;
}

/* Inputs */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, .stTextInput > div > div {
  background: var(--surface) !important; border-color: var(--line) !important; border-radius: 10px !important;
}

/* Buttons (default) */
.stButton > button {
  border-radius:10px !important; border:1px solid var(--line) !important;
  background: var(--surface) !important; color: var(--text) !important;
  font-weight:600 !important;
}
.stButton > button:hover { border-color: var(--line-strong) !important; background: var(--surface-2) !important; }

/* Dataframe — strong borders, larger header, bolder body */
[data-testid="stDataFrame"] {
  border:1px solid var(--line) !important; border-radius: var(--radius) !important;
  overflow:hidden !important; width: 100% !important;
  background: rgba(20, 27, 41, 0.55) !important;
}
[data-testid="stDataFrame"] > div { width: 100% !important; }
[data-testid="stDataFrame"] thead tr th {
  background: rgba(20, 27, 41, 0.95) !important;
  color: #c5cee0 !important;
  font-weight: 700 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] tbody tr td { color: var(--text) !important; font-size: 0.92rem !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background: rgba(106,169,255,0.05) !important; }

/* Match row card */
.match-card {
  display:grid; grid-template-columns: 1.6fr 1fr 1fr;
  gap:14px; align-items:center;
  padding:14px 16px; border-radius: var(--radius);
  border:1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  margin-bottom: 10px;
}
.match-card .meta { color:var(--muted); font-size:0.85rem; }
.match-card .name { color:#fff; font-weight:700; font-size:1.05rem; letter-spacing:-0.01em; }
.match-card .center { text-align:center; }
.match-card .center .vs { color:var(--muted); font-weight:600; letter-spacing:0.06em; font-size:0.78rem; }
.match-card .right { text-align:right; }

.date-group {
  margin: 22px 0 8px 0; padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
  display:flex; justify-content:space-between; align-items:baseline;
}
.date-group .day { color:#fff; font-weight:700; letter-spacing:-0.01em; font-size:1.05rem; }
.date-group .day .dow { color:var(--muted); font-weight:500; margin-right:8px; text-transform:uppercase; letter-spacing:0.05em; font-size:0.78rem; }
.date-group .count { color:var(--muted); font-size:0.85rem; }

.win-pill {
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:0.78rem; font-weight:700;
  background: rgba(33, 194, 133, 0.14); color:#7ee2b1; border:1px solid rgba(33,194,133,0.30);
}
.demo-pill {
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:0.72rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;
  background: rgba(255, 200, 80, 0.10); color:#ffcc66; border:1px solid rgba(255,200,80,0.35);
}
.live-pill {
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:0.72rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;
  background: rgba(33, 194, 133, 0.12); color:#7ee2b1; border:1px solid rgba(33, 194, 133, 0.35);
}

.bar-bg { width:100%; height:8px; background: rgba(255,255,255,0.08); border-radius:4px; overflow:hidden; margin-top:6px; }
.bar-fill { height:100%; border-radius:4px; }
.bar-a { background: linear-gradient(90deg, #ff7059, #ffb27a); }
.bar-b { background: linear-gradient(90deg, #4ea1ff, #7c5cff); }

.empty-state {
  text-align:center; padding: 40px 22px;
  border:1px dashed var(--line); border-radius: var(--radius);
  color: var(--muted);
}

/* Player profile header */
.profile-header { display:flex; gap:18px; align-items:center; padding:18px;
  border:1px solid var(--line); border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  margin-bottom: 14px;
}
.profile-header .avatar img { width:120px; height:120px; border-radius:50%; border:2px solid var(--line-strong); object-fit:cover; background:var(--surface); }
.profile-header .meta-block .name { font-size:1.6rem; font-weight:800; color:#fff; letter-spacing:-0.02em; }
.profile-header .meta-block .sub { color:var(--muted); margin-top:4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# Paths
# =============================================================================

PRED_PATH = PROCESSED_DIR / "all_predictions.csv"
HISTORY_PATH = PROCESSED_DIR / "matches_with_elo_form_sets.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.txt"
ASSETS_DIR = PROJECT_ROOT / "assets"

NAV = ["Matches", "Upcoming", "Players", "Tournaments", "What-if", "Leaderboard"]


# =============================================================================
# Session state
# =============================================================================

def _init_state() -> None:
    st.session_state.setdefault("view", "Matches")
    st.session_state.setdefault("profile_player", None)
    st.session_state.setdefault("profile_tournament", None)


def navigate_to_player(name: str) -> None:
    st.session_state["view"] = "Players"
    st.session_state["profile_player"] = str(name)


def navigate_to_tournament(name: str) -> None:
    st.session_state["view"] = "Tournaments"
    st.session_state["profile_tournament"] = str(name)


def navigate_to_view(view: str) -> None:
    st.session_state["view"] = view


# =============================================================================
# Loaders (cached)
# =============================================================================

def _coerce_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(" ", " ", regex=False)
                .str.strip()
            )
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _coerce_str(df, ["surface", "playerA", "playerB"])
    return df


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
    df = _coerce_str(df, ["surface", "round", "tournament", "playerA", "playerB"])
    return df


@st.cache_data(show_spinner=False)
def load_real_fixtures() -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FIXTURES_PATH)
    if "match_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"match_date": "date"})
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    if "tourney" in df.columns and "tournament" not in df.columns:
        df = df.rename(columns={"tourney": "tournament"})
    df = _coerce_str(df, ["tournament", "surface", "round", "playerA", "playerB"])
    for c in ("oddsA", "oddsB"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, Any, List[str]]:
    if not (MODEL_PATH.exists() and IMPUTER_PATH.exists() and FEATURE_COLS_PATH.exists()):
        return None, None, []
    model = joblib_load(MODEL_PATH)
    imputer = joblib_load(IMPUTER_PATH)
    feature_cols = load_feature_list(FEATURE_COLS_PATH)
    return model, imputer, feature_cols


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def player_directory(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["player", "matches", "wins", "losses", "first_year", "last_year", "winrate"])
    a = history_df.groupby("playerA").agg(wins=("playerA", "size"), first_a=("date", "min"), last_a=("date", "max"))
    b = history_df.groupby("playerB").agg(losses=("playerB", "size"), first_b=("date", "min"), last_b=("date", "max"))
    out = pd.concat([a, b], axis=1).fillna({"wins": 0, "losses": 0})
    out[["wins", "losses"]] = out[["wins", "losses"]].astype(int)
    out["matches"] = out["wins"] + out["losses"]
    out["winrate"] = out["wins"] / out["matches"].replace(0, np.nan)
    first = pd.concat([out["first_a"], out["first_b"]], axis=1).min(axis=1)
    last = pd.concat([out["last_a"], out["last_b"]], axis=1).max(axis=1)
    out["first_year"] = first.dt.year
    out["last_year"] = last.dt.year
    out = out.drop(columns=[c for c in ["first_a", "first_b", "last_a", "last_b"] if c in out.columns])
    out = out.reset_index().rename(columns={"index": "player"})
    if "playerA" in out.columns:
        out = out.rename(columns={"playerA": "player"})
    out = out.sort_values(["matches", "wins"], ascending=[False, False])
    return out


# =============================================================================
# API-Tennis integration
# =============================================================================

def _api_key() -> Optional[str]:
    try_load_dotenv(PROJECT_ROOT)
    key = getenv("API_TENNIS_KEY")
    if not key or "PASTE" in key.upper():
        return None
    return key


@st.cache_data(show_spinner=False, ttl=3600)
def cached_recent_api_fixtures() -> List[Dict[str, Any]]:
    """Pull a wide window of fixtures once for image lookup."""
    if get_fixtures is None or ApiTennisConfig is None:
        return []
    key = _api_key()
    if not key:
        return []
    cfg = ApiTennisConfig(api_key=key, cache_ttl_s=86400)
    today = dt.date.today()
    try:
        return list(get_fixtures(cfg, today - dt.timedelta(days=120), today + dt.timedelta(days=30)))
    except Exception:
        return []


def _name_aliases(name: str) -> set:
    """Aliases used to match a 'Sinner J.' style history name with a 'Jannik
    Sinner' style API name."""
    if not name:
        return set()
    s = name.strip().lower()
    if not s:
        return set()
    aliases = {s}
    norm = re.sub(r"[.,]", " ", s)
    norm = re.sub(r"\s+", " ", norm).strip()
    aliases.add(norm)
    parts = norm.split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        # If "X Y" with last part single letter -> it's surname-initial format.
        if len(last) == 1:
            aliases.add(first)  # surname
            aliases.add(f"{last} {first}")  # "j sinner" -> match API "j sinner"-ish
        else:
            aliases.add(last)  # surname
            if len(first) >= 1:
                aliases.add(f"{last} {first[0]}")  # API to history-style
    return {a for a in aliases if len(a) >= 2}


def _safe_name_match(history_name: str, api_name: str) -> bool:
    h_aliases = _name_aliases(history_name)
    a_aliases = _name_aliases(api_name)
    common = h_aliases & a_aliases
    # Reject single-letter intersections (too noisy).
    return any(len(c) >= 3 for c in common)


def fetch_player_image_via_api(name: str) -> Optional[Path]:
    if not name:
        return None
    fixtures = cached_recent_api_fixtures()
    if not fixtures:
        return None
    target_url: Optional[str] = None
    for ev in fixtures:
        for player_field, logo_field in [
            ("event_first_player", "event_first_player_logo"),
            ("event_second_player", "event_second_player_logo"),
        ]:
            api_name = (ev.get(player_field) or "").strip()
            if not api_name:
                continue
            if _safe_name_match(name, api_name):
                logo = (ev.get(logo_field) or "").strip()
                if logo and logo.startswith("http"):
                    target_url = logo
                    break
        if target_url:
            break
    if not target_url:
        return None

    out = ASSETS_DIR / "players" / f"{slugify(name)}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(target_url, timeout=20)
        r.raise_for_status()
        out.write_bytes(r.content)
        return out
    except Exception:
        return None


def player_image_html(name: str, size: int = 120) -> str:
    """Return an inline <img> tag — local cache first, otherwise SVG avatar."""
    p = ASSETS_DIR / "players" / slugify(name)
    img = find_image(p)
    if img and img.exists():
        try:
            data = base64.b64encode(img.read_bytes()).decode()
            ext = img.suffix.lower().lstrip(".")
            mime = "jpeg" if ext == "jpg" else ext
            src = f"data:image/{mime};base64,{data}"
            return (
                f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
                f'overflow:hidden;border:2px solid var(--line-strong);background:var(--surface);">'
                f'<img src="{src}" width="{size}" height="{size}" '
                f'style="width:100%;height:100%;object-fit:cover;display:block;"/></div>'
            )
        except Exception:
            pass
    return (
        f'<img src="{svg_avatar_data_uri(name, size)}" width="{size}" height="{size}" '
        f'style="border-radius:50%;border:2px solid var(--line-strong);"/>'
    )


def run_real_upcoming_fetch() -> Tuple[bool, str]:
    cmds = [
        [sys.executable, "-m", "src.data.fetch_upcoming_apitennis"],
        [sys.executable, "-m", "src.data.fetch_odds_apitennis"],
    ]
    out: List[str] = []
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
            out.append("$ " + " ".join(cmd))
            if p.stdout:
                out.append(p.stdout.strip())
            if p.stderr:
                out.append(p.stderr.strip())
            if p.returncode != 0:
                return False, "\n".join([x for x in out if x])
        except Exception as e:
            out.append(str(e))
            return False, "\n".join([x for x in out if x])
    return True, "\n".join([x for x in out if x])


# =============================================================================
# Helpers
# =============================================================================

def _fmt_pct(x: Optional[float], n: int = 1) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x*100:.{n}f}%"


def _fmt_num(x: Optional[float], n: int = 4) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.{n}f}"


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="ps-kpi-sub">{h(sub)}</div>' if sub else ""
    return (
        f'<div class="ps-kpi">'
        f'<div class="ps-kpi-label">{h(label)}</div>'
        f'<div class="ps-kpi-val">{h(value)}</div>'
        f"{sub_html}"
        f"</div>"
    )


def confidence_label(p: float) -> str:
    d = abs(p - 0.5)
    if d < 0.06:
        return "Low"
    if d < 0.14:
        return "Medium"
    return "High"


def all_tournaments(df: pd.DataFrame) -> List[str]:
    if df.empty or "tournament" not in df.columns:
        return []
    s = df["tournament"].dropna().astype(str).map(lambda x: x.strip())
    s = s[s != ""]
    return sorted(s.unique().tolist())


def render_nav() -> None:
    info = load_metrics_json()
    model_label = str(info.get("model", "Model"))
    val = info.get("validation") or {}
    pill = "Live model"
    if isinstance(val.get("logloss"), (int, float)) and isinstance(val.get("accuracy"), (int, float)):
        pill = f"{model_label} | acc {val['accuracy']*100:.1f}% | log loss {val['logloss']:.3f}"
    st.markdown(
        f"""
        <div class="ps-nav">
          <div class="ps-nav-left">
            <div class="ps-logo">PS</div>
            <div class="ps-brand">Predictive Serve<small>Tennis forecasting console</small></div>
          </div>
          <span class="ps-pill"><span class="dot"></span>{h(pill)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="ps-hero">
          <span class="ps-eyebrow">Predictive Serve · ATP forecasting</span>
          <h1><span class="grad">Match-level forecasts, end-to-end.</span></h1>
          <p>Historical pipeline 2000-present (67k+ matches), Elo + form + head-to-head features, gradient-boosted model with a held-out 2025 test split, and live ATP fixtures with player photos via API-Tennis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    if df is None or df.empty or "p_model" not in df.columns or "y" not in df.columns:
        st.markdown('<div class="empty-state">No predictions available yet. Run the pipeline first.</div>', unsafe_allow_html=True)
        return
    sub = df.dropna(subset=["p_model", "y"]).copy()
    if sub.empty:
        st.markdown('<div class="empty-state">No scored matches in the current selection.</div>', unsafe_allow_html=True)
        return
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    y = sub["y"].astype(int).values
    p = sub["p_model"].astype(float).values
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    ac = accuracy_score(y, (p >= 0.5).astype(int))
    high_mask = (p < 0.35) | (p > 0.65)
    high_n = int(high_mask.sum())
    high_acc = (
        accuracy_score(y[high_mask], (p[high_mask] >= 0.5).astype(int))
        if high_n > 0
        else None
    )

    tiles = [
        _kpi("Predictions", f"{len(sub):,}", f"date range {sub['date'].min().date()} - {sub['date'].max().date()}"),
        _kpi("Pick accuracy", _fmt_pct(ac), "share of correct AI picks"),
        _kpi("Log loss", _fmt_num(ll), "lower is better"),
        _kpi("High-confidence accuracy", _fmt_pct(high_acc), f"on {high_n:,} confident calls"),
    ]
    st.markdown(f'<div class="ps-kpi-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_top_nav_buttons() -> None:
    """Tab-like button row that drives st.session_state['view']."""
    st.markdown('<div class="ps-tabs">', unsafe_allow_html=True)
    cols = st.columns(len(NAV))
    active = st.session_state.get("view", "Matches")
    for i, name in enumerate(NAV):
        is_active = name == active
        kind = "primary" if is_active else "secondary"
        cols[i].button(
            name,
            key=f"nav_btn_{name}",
            type=kind,
            on_click=navigate_to_view,
            args=(name,),
            width="stretch",
        )
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# Selection-driven action buttons
# =============================================================================

def _render_action_buttons(prefix: str, player_a: Optional[str], player_b: Optional[str], tournament: Optional[str]) -> None:
    """Buttons that navigate to a profile based on a selected row."""
    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        if player_a:
            st.button(
                f"Open: {player_a}",
                key=f"{prefix}_pa",
                on_click=navigate_to_player,
                args=(player_a,),
                width="stretch",
            )
    with cols[1]:
        if player_b:
            st.button(
                f"Open: {player_b}",
                key=f"{prefix}_pb",
                on_click=navigate_to_player,
                args=(player_b,),
                width="stretch",
            )
    with cols[2]:
        if tournament:
            st.button(
                f"Open: {tournament}",
                key=f"{prefix}_t",
                on_click=navigate_to_tournament,
                args=(tournament,),
                width="stretch",
            )


# =============================================================================
# Tab: Matches
# =============================================================================

def _style_correct_rows(df_for_style: pd.DataFrame, correct_mask: np.ndarray):
    """Apply pastel green / red row backgrounds based on correctness, plus
    column-level emphasis on the AI Pick / Confidence cells."""
    bg_correct = "background-color: rgba(45, 210, 154, 0.13);"
    bg_wrong = "background-color: rgba(255, 100, 113, 0.13);"

    def _row_color(row):
        try:
            i = df_for_style.index.get_loc(row.name)
        except Exception:
            return [""] * len(row)
        if i >= len(correct_mask):
            return [""] * len(row)
        return [bg_correct if correct_mask[i] else bg_wrong] * len(row)

    styler = df_for_style.style.apply(_row_color, axis=1)

    # Bold the columns that drive interpretation
    bold_cols = [c for c in ("AI Pick", "Actual Winner", "Result") if c in df_for_style.columns]
    if bold_cols:
        styler = styler.set_properties(subset=bold_cols, **{"font-weight": "700", "color": "#ffffff"})
    if "Confidence" in df_for_style.columns:
        styler = styler.set_properties(
            subset=["Confidence"],
            **{"font-weight": "700", "color": "#9bc7ff"},
        )
    return styler


def tab_matches(pred_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Matches Explorer</div>", unsafe_allow_html=True)
    if pred_df.empty:
        st.markdown(
            '<div class="empty-state">No predictions found. Run <code>py -m src.models.score_all_matches</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    surf_opts = sorted(pred_df["surface"].dropna().unique().tolist()) if "surface" in pred_df.columns else []
    min_d = pd.to_datetime(pred_df["date"].min()).date()
    max_d = pd.to_datetime(pred_df["date"].max()).date()

    f1, f2, f3 = st.columns([2.4, 2, 1.5])
    with f1:
        dr = st.date_input("Date range", value=(max_d - pd.Timedelta(days=180), max_d), key="m_dates")
        if isinstance(dr, (list, tuple)):
            if len(dr) == 2:
                d1, d2 = dr
            elif len(dr) == 1:
                d1 = d2 = dr[0]
            else:
                d1, d2 = min_d, max_d
        else:
            d1 = d2 = dr
        if d1 is None or d2 is None:
            d1, d2 = min_d, max_d
    with f2:
        sel_surfaces = st.multiselect("Surface", surf_opts, default=surf_opts, key="m_surfaces")
    with f3:
        min_conf = st.slider("Minimum AI confidence", 0.50, 0.95, 0.50, 0.01, key="m_minconf")

    f4, f5 = st.columns([2.5, 2])
    with f4:
        name_filter = st.text_input("Player contains", value="", placeholder="e.g. Sinner", key="m_name").strip().lower()
    with f5:
        result_filter = st.selectbox(
            "Outcome filter",
            ["All", "AI was correct", "AI was wrong"],
            index=0,
            key="m_result",
        )

    df = pred_df.copy()
    df = df[(df["date"].dt.date >= d1) & (df["date"].dt.date <= d2)]
    if sel_surfaces:
        df = df[df["surface"].isin(sel_surfaces)]
    df = df.dropna(subset=["p_model"])
    df["winner_prob"] = np.where(df["p_model"] >= 0.5, df["p_model"], 1.0 - df["p_model"])
    df["winner_pick"] = np.where(df["p_model"] >= 0.5, df["playerA"], df["playerB"])
    df = df[df["winner_prob"] >= min_conf]
    if name_filter:
        m = (
            df["playerA"].astype(str).str.lower().str.contains(name_filter, na=False)
            | df["playerB"].astype(str).str.lower().str.contains(name_filter, na=False)
        )
        df = df[m]

    actual_winner = (
        np.where(df["y"].astype("Int64").fillna(-1).astype(int) == 1, df["playerA"], df["playerB"])
        if "y" in df.columns
        else None
    )
    if actual_winner is not None and result_filter != "All":
        correct = df["winner_pick"].values == actual_winner
        df = df[correct] if result_filter == "AI was correct" else df[~correct]
        actual_winner = (
            np.where(df["y"].astype("Int64").fillna(-1).astype(int) == 1, df["playerA"], df["playerB"])
            if "y" in df.columns
            else None
        )

    render_kpis(df)
    st.markdown(
        f"<div class='ps-section-title'>Results &middot; {len(df):,} matches</div>",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.markdown('<div class="empty-state">No matches match the current filters.</div>', unsafe_allow_html=True)
        return

    df = df.sort_values("date", ascending=False).head(500)
    actual_winner = (
        np.where(df["y"].astype("Int64").fillna(-1).astype(int) == 1, df["playerA"], df["playerB"])
        if "y" in df.columns
        else None
    )
    correct_mask = (df["winner_pick"].values == actual_winner) if actual_winner is not None else None

    show = pd.DataFrame({
        "Date": df["date"].dt.strftime("%Y-%m-%d"),
        "Surface": df["surface"],
        "Player A": df["playerA"],
        "Player B": df["playerB"],
        "AI Pick": df["winner_pick"],
        "Confidence": df["winner_prob"] * 100.0,
    })
    if actual_winner is not None:
        show["Actual Winner"] = actual_winner
        show["Result"] = np.where(correct_mask, "✓", "✗")

    if correct_mask is not None:
        styled = _style_correct_rows(show, correct_mask).format({"Confidence": "{:.1f}%"})
        st.dataframe(
            styled,
            width="stretch",
            hide_index=False,
            height=520,
        )
    else:
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            height=520,
            column_config={"Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%")},
        )

    # Quick navigation: pick a row from the underlying df via select widgets
    st.caption("Tip: pick any player / tournament below to open their profile.")
    sel_cols = st.columns([2, 2, 2, 1.5])
    with sel_cols[0]:
        a_sel = st.selectbox(
            "Player A",
            ["—"] + sorted(df["playerA"].dropna().unique().tolist()),
            key="m_jump_a",
        )
        if a_sel != "—":
            st.button(f"Open profile: {a_sel}", key="m_open_a", on_click=navigate_to_player, args=(a_sel,), width="stretch")
    with sel_cols[1]:
        b_sel = st.selectbox(
            "Player B",
            ["—"] + sorted(df["playerB"].dropna().unique().tolist()),
            key="m_jump_b",
        )
        if b_sel != "—":
            st.button(f"Open profile: {b_sel}", key="m_open_b", on_click=navigate_to_player, args=(b_sel,), width="stretch")
    with sel_cols[2]:
        if "tournament" in df.columns:
            t_sel = st.selectbox(
                "Tournament",
                ["—"] + sorted(df["tournament"].dropna().unique().tolist()),
                key="m_jump_t",
            )
            if t_sel != "—":
                st.button(f"Open: {t_sel}", key="m_open_t", on_click=navigate_to_tournament, args=(t_sel,), width="stretch")


# =============================================================================
# Tab: Upcoming
# =============================================================================

_TOURNEY_BY_MONTH = {
    1: [("Australian Open", "Hard"), ("Adelaide", "Hard"), ("United Cup", "Hard")],
    2: [("Rotterdam", "Hard"), ("Dubai", "Hard"), ("Acapulco", "Hard")],
    3: [("Indian Wells", "Hard"), ("Miami Open", "Hard")],
    4: [("Monte Carlo", "Clay"), ("Barcelona", "Clay"), ("Estoril", "Clay")],
    5: [("Madrid", "Clay"), ("Rome", "Clay"), ("Roland Garros", "Clay")],
    6: [("Roland Garros", "Clay"), ("Queen's Club", "Grass"), ("Halle", "Grass")],
    7: [("Wimbledon", "Grass"), ("Hamburg", "Clay"), ("Newport", "Grass")],
    8: [("Canadian Open", "Hard"), ("Cincinnati", "Hard"), ("US Open", "Hard")],
    9: [("US Open", "Hard"), ("Laver Cup", "Hard")],
    10: [("Shanghai", "Hard"), ("Vienna", "Hard"), ("Stockholm", "Hard")],
    11: [("Paris Masters", "Hard"), ("ATP Finals", "Hard")],
    12: [("Davis Cup Finals", "Hard"), ("United Cup", "Hard")],
}
_ROUNDS = ["R32", "R32", "R16", "R16", "R16", "QF", "QF", "SF", "F"]


@st.cache_data(show_spinner=False)
def synth_fixtures(history_df: pd.DataFrame, days_ahead: int = 14, n_matches: int = 60, seed: int = 7) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    cutoff = history_df["date"].max() - pd.Timedelta(days=365)
    recent = history_df[history_df["date"] >= cutoff]
    if recent.empty:
        recent = history_df.tail(20_000)
    counts = pd.concat([recent["playerA"], recent["playerB"]]).value_counts()
    pool = counts.head(48).index.tolist()
    if len(pool) < 6:
        return pd.DataFrame()
    rng = random.Random(seed)
    today = pd.Timestamp.today().normalize()
    rows = []
    for i in range(n_matches):
        a, b = rng.sample(pool, 2)
        day_offset = rng.randint(0, max(1, days_ahead - 1))
        match_date = today + pd.Timedelta(days=day_offset)
        month = match_date.month
        tour, surface = rng.choice(_TOURNEY_BY_MONTH.get(month, [("Tour Event", "Hard")]))
        rnd = rng.choice(_ROUNDS)
        rows.append({
            "match_id": f"demo-{i:04d}",
            "date": match_date,
            "tournament": tour,
            "surface": surface,
            "round": rnd,
            "playerA": a,
            "playerB": b,
            "oddsA": np.nan,
            "oddsB": np.nan,
            "_source": "demo",
        })
    df = pd.DataFrame(rows).sort_values(["date", "tournament", "playerA"])
    return df


def _score_fixtures(fix_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    model, imputer, feature_cols = load_artifacts()
    if model is None or imputer is None or not feature_cols or build_feature_row is None:
        return pd.DataFrame()
    rows = []
    for _, r in fix_df.iterrows():
        try:
            row_df = build_feature_row(
                history=history_df,
                feature_cols=feature_cols,
                playerA=str(r["playerA"]),
                playerB=str(r["playerB"]),
                surface=str(r["surface"]),
                date=pd.Timestamp(r["date"]),
                round_code=str(r.get("round") or "") or None,
                oddsA=float(r["oddsA"]) if pd.notna(r.get("oddsA")) else None,
                oddsB=float(r["oddsB"]) if pd.notna(r.get("oddsB")) else None,
            )
            X = imputer.transform(row_df[feature_cols])
            p = float(model.predict_proba(X)[0, 1])
        except Exception:
            continue
        rows.append({
            "date": pd.Timestamp(r["date"]),
            "tournament": r.get("tournament", ""),
            "round": r.get("round", ""),
            "surface": r["surface"],
            "playerA": r["playerA"],
            "playerB": r["playerB"],
            "p_model": p,
            "_source": r.get("_source", "real"),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["winner_prob"] = np.where(out["p_model"] >= 0.5, out["p_model"], 1.0 - out["p_model"])
    out["winner_pick"] = np.where(out["p_model"] >= 0.5, out["playerA"], out["playerB"])
    return out


def tab_upcoming(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Upcoming Fixtures</div>", unsafe_allow_html=True)

    real = load_real_fixtures()
    has_real = not real.empty
    has_key = _api_key() is not None

    if has_real:
        # show only future
        real = real.copy()
        real["date"] = pd.to_datetime(real["date"], errors="coerce")
        real = real.dropna(subset=["date", "playerA", "playerB", "surface"])
        real = real[real["date"] >= pd.Timestamp.today().normalize()]
        if real.empty:
            has_real = False

    # Toolbar: API key status + refresh
    bar1, bar2 = st.columns([3, 1.6])
    with bar1:
        if has_real:
            st.markdown('<span class="live-pill">live fixtures</span>', unsafe_allow_html=True)
            st.caption(f"Loaded from `{FIXTURES_PATH}`")
        elif has_key:
            st.markdown('<span class="demo-pill">demo data</span>', unsafe_allow_html=True)
            st.caption("API key detected — click the refresh button to pull live ATP fixtures.")
        else:
            st.markdown('<span class="demo-pill">demo data &middot; no api key</span>', unsafe_allow_html=True)
            st.caption(
                "Create a `.env` file in the project root with "
                "`API_TENNIS_KEY=<your key>` to enable live fixtures."
            )
    with bar2:
        disabled = not has_key
        if st.button("Refresh fixtures from API", key="u_refresh", disabled=disabled, width="stretch"):
            with st.spinner("Calling api-tennis.com..."):
                ok, log_text = run_real_upcoming_fetch()
                load_real_fixtures.clear()
            if ok:
                st.success("Fixtures updated.")
            else:
                st.error("Refresh failed.")
            with st.expander("Refresh log", expanded=not ok):
                st.code(log_text or "(empty)", language="text")
            st.rerun()

    # Choose data source
    if has_real:
        fix = real
    else:
        fix = synth_fixtures(history_df)

    if fix.empty:
        st.markdown('<div class="empty-state">Could not generate any fixtures (history dataset empty).</div>', unsafe_allow_html=True)
        return

    available_days = sorted(fix["date"].dt.date.unique().tolist())
    day_labels = [d.strftime("%a %b %d, %Y") for d in available_days]

    f1, f2, f3, f4 = st.columns([1.6, 2, 2, 1.2])
    with f1:
        day_choice = st.selectbox("Day", ["All days"] + day_labels, index=0, key="u_day")
    with f2:
        tour_opts = all_tournaments(fix)
        sel_tours = st.multiselect("Tournament", tour_opts, default=[], key="u_tours")
    with f3:
        surf_opts = sorted(fix["surface"].dropna().unique().tolist())
        sel_surfaces = st.multiselect("Surface", surf_opts, default=surf_opts, key="u_surfaces")
    with f4:
        min_conf = st.slider("Min confidence", 0.50, 0.95, 0.55, 0.01, key="u_minconf")

    if day_choice != "All days":
        idx = day_labels.index(day_choice)
        fix = fix[fix["date"].dt.date == available_days[idx]]
    if sel_tours:
        fix = fix[fix["tournament"].isin(sel_tours)]
    if sel_surfaces:
        fix = fix[fix["surface"].isin(sel_surfaces)]
    if fix.empty:
        st.markdown('<div class="empty-state">No fixtures match the current filters.</div>', unsafe_allow_html=True)
        return

    sc = _score_fixtures(fix, history_df)
    if sc.empty:
        st.markdown(
            '<div class="empty-state">Could not score fixtures (model artifacts or history snapshots missing).</div>',
            unsafe_allow_html=True,
        )
        return

    sc = sc[sc["winner_prob"] >= min_conf].sort_values(["date", "winner_prob"], ascending=[True, False])

    st.markdown(
        f"<div class='ps-section-title'>{len(sc):,} matches scored</div>",
        unsafe_allow_html=True,
    )

    for day, group in sc.groupby(sc["date"].dt.date):
        day_dt = pd.Timestamp(day)
        st.markdown(
            f"""
            <div class="date-group">
              <div class="day"><span class="dow">{day_dt.strftime('%a')}</span>{day_dt.strftime('%B %d, %Y')}</div>
              <div class="count">{len(group)} match{'es' if len(group) != 1 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for i, (_, r) in enumerate(group.iterrows()):
            meta_bits = []
            if r.get("tournament"):
                meta_bits.append(str(r["tournament"]))
            if r.get("surface"):
                meta_bits.append(str(r["surface"]))
            if r.get("round"):
                meta_bits.append(str(r["round"]))
            meta = " · ".join(meta_bits)

            st.markdown(
                f"""
                <div class="match-card">
                  <div>
                    <div class="meta">{h(meta)}</div>
                    <div class="name">{h(str(r['playerA']))} <span style="opacity:.5;font-weight:500;">vs</span> {h(str(r['playerB']))}</div>
                  </div>
                  <div class="center">
                    <div class="vs">AI PICK</div>
                    <div class="name" style="margin-top:4px;">{h(str(r['winner_pick']))}</div>
                    <div class="meta">{confidence_label(float(r['winner_prob']))} confidence</div>
                  </div>
                  <div class="right">
                    <div class="win-pill">{r['winner_prob']*100:.1f}%</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uid = f"u_{day}_{i}"
            _render_action_buttons(
                prefix=uid,
                player_a=str(r["playerA"]),
                player_b=str(r["playerB"]),
                tournament=str(r.get("tournament") or "") or None,
            )


# =============================================================================
# Tab: Players
# =============================================================================

def _player_history(history_df: pd.DataFrame, player: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    mask = (history_df["playerA"] == player) | (history_df["playerB"] == player)
    h_ = history_df[mask].copy().sort_values("date")
    if h_.empty:
        return h_
    h_["is_winner"] = (h_["playerA"] == player).astype(int)
    h_["opponent"] = np.where(h_["is_winner"] == 1, h_["playerB"], h_["playerA"])
    if "eloA" in h_.columns and "eloB" in h_.columns:
        h_["elo_player"] = np.where(h_["is_winner"] == 1, h_["eloA"], h_["eloB"])
    return h_


def tab_players(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Player Profiles</div>", unsafe_allow_html=True)
    if history_df.empty:
        st.markdown('<div class="empty-state">No history dataset available.</div>', unsafe_allow_html=True)
        return

    directory = player_directory(history_df)
    if directory.empty:
        st.markdown('<div class="empty-state">Player directory is empty.</div>', unsafe_allow_html=True)
        return

    year_min = int(directory["first_year"].dropna().min())
    year_max = int(directory["last_year"].dropna().max())

    f1, f2, f3 = st.columns([2.2, 1.5, 2.5])
    with f1:
        search = st.text_input("Search by name", value="", placeholder="e.g. Federer", key="p_search").strip().lower()
    with f2:
        min_matches = st.number_input("Min career matches", min_value=10, max_value=2000, value=80, step=10, key="p_min")
    with f3:
        if year_min < year_max:
            year_range = st.slider(
                "Active during",
                min_value=year_min,
                max_value=year_max,
                value=(max(year_min, year_max - 10), year_max),
                key="p_years",
            )
        else:
            year_range = (year_min, year_max)

    flt = directory[directory["matches"] >= int(min_matches)]
    flt = flt[~((flt["last_year"] < year_range[0]) | (flt["first_year"] > year_range[1]))]
    if search:
        flt = flt[flt["player"].astype(str).str.lower().str.contains(search, na=False)]
    flt = flt.sort_values(["matches", "wins"], ascending=[False, False])

    if flt.empty:
        st.markdown('<div class="empty-state">No players match the current filters.</div>', unsafe_allow_html=True)
        return

    options = flt.head(400).copy()
    label_for = {
        row["player"]: f"{row['player']}  -  {int(row['matches']):,} matches  ·  {(row['winrate']*100):.1f}% WR  ·  {int(row['first_year'])}-{int(row['last_year'])}"
        for _, row in options.iterrows()
    }
    keys = list(options["player"])
    pre = st.session_state.get("profile_player")
    default_idx = keys.index(pre) if pre in keys else 0

    player = st.selectbox(
        f"Select a player ({len(flt):,} match the filters, top 400 shown)",
        keys,
        index=default_idx,
        format_func=lambda k: label_for.get(k, k),
        key="p_select",
    )
    if not player:
        return
    st.session_state["profile_player"] = player

    h_ = _player_history(history_df, player)
    if h_.empty:
        st.markdown('<div class="empty-state">No matches found for this player.</div>', unsafe_allow_html=True)
        return

    total = len(h_)
    wins = int(h_["is_winner"].sum())
    losses = total - wins
    winrate = wins / total if total else 0.0
    last_year_window = h_[h_["date"] >= (h_["date"].max() - pd.Timedelta(days=365))]
    recent_wr = last_year_window["is_winner"].mean() if len(last_year_window) else None
    first_season = str(h_["date"].min().year)
    last_season = str(h_["date"].max().year)

    img_html = player_image_html(player, size=120)
    st.markdown(
        f"""
        <div class="profile-header">
          <div class="avatar">{img_html}</div>
          <div class="meta-block">
            <div class="name">{h(player)}</div>
            <div class="sub">Career: {h(first_season)}-{h(last_season)} &middot; {total:,} matches &middot; {wins:,}-{losses:,} W-L &middot; {winrate*100:.1f}% WR</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Image fetch action
    has_local_img = find_image(ASSETS_DIR / "players" / slugify(player)) is not None
    api_present = _api_key() is not None
    img_cols = st.columns([1.4, 1.4, 4])
    with img_cols[0]:
        if api_present:
            if st.button("Fetch live photo", key="p_fetch_img", width="stretch"):
                with st.spinner("Searching API-Tennis fixtures for this player..."):
                    saved = fetch_player_image_via_api(player)
                if saved is not None:
                    st.success("Photo updated.")
                    st.rerun()
                else:
                    st.warning("No matching player photo in recent fixtures.")
        else:
            st.button("Fetch live photo", key="p_fetch_img_disabled", disabled=True, width="stretch")
    with img_cols[1]:
        if has_local_img:
            if st.button("Reset to avatar", key="p_reset_img", width="stretch"):
                try:
                    img_path = find_image(ASSETS_DIR / "players" / slugify(player))
                    if img_path and img_path.exists():
                        img_path.unlink()
                    st.rerun()
                except Exception:
                    pass

    tiles = (
        _kpi("Matches", f"{total:,}")
        + _kpi("Wins / Losses", f"{wins:,} - {losses:,}")
        + _kpi("Career WR", _fmt_pct(winrate))
        + _kpi("Last 365d WR", _fmt_pct(recent_wr))
    )
    st.markdown(f'<div class="ps-kpi-grid">{tiles}</div>', unsafe_allow_html=True)

    if "elo_player" in h_.columns:
        st.markdown("<div class='ps-section-title'>Elo trajectory</div>", unsafe_allow_html=True)
        elo_df = h_[["date", "elo_player", "surface"]].dropna(subset=["elo_player", "date"]).copy()
        if not elo_df.empty:
            chart = (
                alt.Chart(elo_df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("elo_player:Q", title="Elo", scale=alt.Scale(zero=False)),
                    color=alt.Color("surface:N", legend=alt.Legend(title="Surface", orient="top")),
                    tooltip=["date:T", "elo_player:Q", "surface:N"],
                )
                .properties(height=300)
                .configure_axis(grid=True, gridColor="#1a2030", labelColor="#9aa3b6", titleColor="#9aa3b6")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, width="stretch")

    surfaces = h_.groupby("surface")["is_winner"].agg(["sum", "count"])
    surfaces["winrate"] = surfaces["sum"] / surfaces["count"]
    if not surfaces.empty:
        st.markdown("<div class='ps-section-title'>Surface breakdown</div>", unsafe_allow_html=True)
        surf_view = surfaces.reset_index().rename(columns={"sum": "Wins", "count": "Matches", "winrate": "Win rate"})
        surf_view["Win rate"] = surf_view["Win rate"] * 100
        st.dataframe(
            surf_view,
            width="stretch",
            hide_index=True,
            column_config={"Win rate": st.column_config.NumberColumn("Win rate", format="%.1f%%")},
        )

    st.markdown("<div class='ps-section-title'>Recent matches</div>", unsafe_allow_html=True)
    recent = h_.sort_values("date", ascending=False).head(20).copy()
    recent_view = pd.DataFrame({
        "Date": recent["date"].dt.strftime("%Y-%m-%d"),
        "Tournament": recent.get("tournament", ""),
        "Surface": recent["surface"],
        "Round": recent.get("round", ""),
        "Result": np.where(recent["is_winner"] == 1, "Win", "Loss"),
        "Opponent": recent["opponent"],
        "Score": recent.get("score", ""),
    })
    st.dataframe(recent_view, width="stretch", hide_index=True, height=380)

    # Quick navigation: opponents & tournaments from this profile
    st.caption("Open a profile from this player's history:")
    nav_cols = st.columns([2, 2])
    with nav_cols[0]:
        opp_options = ["—"] + sorted(recent["opponent"].dropna().unique().tolist())
        opp_pick = st.selectbox("Recent opponent", opp_options, key="p_jump_opp")
        if opp_pick != "—":
            st.button(f"Open profile: {opp_pick}", key="p_open_opp", on_click=navigate_to_player, args=(opp_pick,), width="stretch")
    with nav_cols[1]:
        if "tournament" in recent.columns:
            t_options = ["—"] + sorted(recent["tournament"].dropna().unique().tolist())
            t_pick = st.selectbox("Recent tournament", t_options, key="p_jump_tour")
            if t_pick != "—":
                st.button(f"Open: {t_pick}", key="p_open_tour", on_click=navigate_to_tournament, args=(t_pick,), width="stretch")


# =============================================================================
# Tab: Tournaments
# =============================================================================

def tab_tournaments(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Tournament Profiles</div>", unsafe_allow_html=True)
    if history_df.empty or "tournament" not in history_df.columns:
        st.markdown('<div class="empty-state">No tournament data available.</div>', unsafe_allow_html=True)
        return

    counts = (
        history_df.groupby("tournament")["date"]
        .agg(["count", "min", "max"])
        .rename(columns={"count": "matches", "min": "first", "max": "last"})
        .reset_index()
    )
    counts["first_year"] = counts["first"].dt.year
    counts["last_year"] = counts["last"].dt.year
    counts = counts.sort_values("matches", ascending=False)

    f1, f2, f3 = st.columns([2.4, 1.6, 2])
    with f1:
        search = st.text_input("Search tournament", value="", placeholder="e.g. Wimbledon", key="t_search").strip().lower()
    with f2:
        min_matches = st.number_input("Min matches recorded", min_value=10, max_value=2000, value=40, step=10, key="t_min")
    with f3:
        year_min = int(counts["first_year"].min())
        year_max = int(counts["last_year"].max())
        if year_min < year_max:
            year_range = st.slider(
                "Active during",
                min_value=year_min,
                max_value=year_max,
                value=(max(year_min, year_max - 10), year_max),
                key="t_years",
            )
        else:
            year_range = (year_min, year_max)

    flt = counts[counts["matches"] >= int(min_matches)]
    flt = flt[~((flt["last_year"] < year_range[0]) | (flt["first_year"] > year_range[1]))]
    if search:
        flt = flt[flt["tournament"].astype(str).str.lower().str.contains(search, na=False)]
    if flt.empty:
        st.markdown('<div class="empty-state">No tournaments match the current filters.</div>', unsafe_allow_html=True)
        return

    label_for = {
        row["tournament"]: f"{row['tournament']}  -  {int(row['matches']):,} matches  ·  {int(row['first_year'])}-{int(row['last_year'])}"
        for _, row in flt.iterrows()
    }
    keys = list(flt["tournament"])
    pre = st.session_state.get("profile_tournament")
    default_idx = keys.index(pre) if pre in keys else 0
    tour = st.selectbox(
        f"Select a tournament ({len(flt):,} match the filters)",
        keys,
        index=default_idx,
        format_func=lambda k: label_for.get(k, k),
        key="t_select",
    )
    if not tour:
        return
    st.session_state["profile_tournament"] = tour

    sub = history_df[history_df["tournament"] == tour].copy()
    if sub.empty:
        st.markdown('<div class="empty-state">No matches found for this tournament.</div>', unsafe_allow_html=True)
        return

    years = sorted(sub["date"].dt.year.dropna().unique().tolist())
    surfaces = sorted(sub["surface"].dropna().unique().tolist())
    champ_per_year: List[Tuple[int, str]] = []
    for y in years:
        round_col = sub.get("round", pd.Series("", index=sub.index)).astype(str).str.upper()
        f_ = sub[(sub["date"].dt.year == y) & (round_col == "F")]
        if not f_.empty:
            champ_per_year.append((int(y), str(f_.iloc[-1]["playerA"])))

    editions = f"{len(years):,}"
    total_matches = f"{len(sub):,}"
    surfaces_label = ", ".join(surfaces) if surfaces else "-"
    first_season = str(min(years)) if years else "-"
    latest_season = str(max(years)) if years else "-"
    players_seen = sub[["playerA", "playerB"]].stack().nunique()
    players_seen_label = f"{players_seen:,}"

    tiles_html = (
        _kpi("Editions", editions)
        + _kpi("Total matches", total_matches)
        + _kpi("Surfaces", surfaces_label)
        + _kpi("First season", first_season)
        + _kpi("Latest season", latest_season)
        + _kpi("Players seen", players_seen_label)
    )
    st.markdown(
        f'<div class="ps-kpi-grid" style="grid-template-columns:repeat(6,minmax(0,1fr));">{tiles_html}</div>',
        unsafe_allow_html=True,
    )

    if champ_per_year:
        st.markdown("<div class='ps-section-title'>Recent champions</div>", unsafe_allow_html=True)
        champ_df = pd.DataFrame(champ_per_year[-15:], columns=["Year", "Champion"])
        st.dataframe(champ_df.iloc[::-1], width="stretch", hide_index=True)

        # quick jump for champions
        st.caption("Open a champion's profile:")
        c_pick = st.selectbox("Champion", ["—"] + champ_df["Champion"].tolist(), key="t_jump_champ")
        if c_pick != "—":
            st.button(f"Open profile: {c_pick}", key="t_open_champ", on_click=navigate_to_player, args=(c_pick,), width="stretch")

    st.markdown("<div class='ps-section-title'>Most recent matches</div>", unsafe_allow_html=True)
    recent = sub.sort_values("date", ascending=False).head(25).copy()
    recent_view = pd.DataFrame({
        "Date": recent["date"].dt.strftime("%Y-%m-%d"),
        "Round": recent.get("round", ""),
        "Surface": recent["surface"],
        "Winner": recent["playerA"],
        "Loser": recent["playerB"],
        "Score": recent.get("score", ""),
    })
    st.dataframe(recent_view, width="stretch", hide_index=True, height=380)

    st.caption("Open a player from this tournament:")
    plr_options = ["—"] + sorted(pd.concat([recent["playerA"], recent["playerB"]]).dropna().astype(str).unique().tolist())
    plr_pick = st.selectbox("Player", plr_options, key="t_jump_player")
    if plr_pick != "—":
        st.button(f"Open profile: {plr_pick}", key="t_open_player", on_click=navigate_to_player, args=(plr_pick,), width="stretch")


# =============================================================================
# Tab: What-if
# =============================================================================

def tab_whatif(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Match Predictor</div>", unsafe_allow_html=True)
    model, imputer, feature_cols = load_artifacts()
    if model is None or build_feature_row is None or history_df.empty:
        st.markdown('<div class="empty-state">Model artifacts or history dataset are missing.</div>', unsafe_allow_html=True)
        return

    directory = player_directory(history_df)
    if directory.empty:
        st.markdown('<div class="empty-state">No players available.</div>', unsafe_allow_html=True)
        return

    surfaces = sorted(history_df["surface"].dropna().unique().tolist())
    rounds = ["", "F", "SF", "QF", "R16", "R32", "R64", "R128", "RR"]

    keys = list(directory["player"])
    label_for = {
        row["player"]: f"{row['player']}  -  {int(row['matches']):,} matches  ·  {(row['winrate']*100):.1f}% WR"
        for _, row in directory.iterrows()
    }

    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    with f1:
        pa = st.selectbox("Player A", keys, index=0, format_func=lambda k: label_for.get(k, k), key="w_pa")
    with f2:
        pb = st.selectbox("Player B", keys, index=min(1, len(keys) - 1), format_func=lambda k: label_for.get(k, k), key="w_pb")
    with f3:
        surface = st.selectbox(
            "Surface",
            surfaces,
            index=surfaces.index("Hard") if "Hard" in surfaces else 0,
            key="w_surface",
        )
    with f4:
        round_code = st.selectbox("Round", rounds, index=0, key="w_round")

    match_date = st.date_input(
        "Match date",
        value=history_df["date"].max().date() if not history_df.empty else pd.Timestamp.today().date(),
        key="w_date",
    )
    if pa == pb:
        st.warning("Pick two different players.")
        return

    try:
        row_df = build_feature_row(
            history=history_df,
            feature_cols=feature_cols,
            playerA=pa,
            playerB=pb,
            surface=str(surface),
            date=pd.Timestamp(match_date),
            round_code=round_code or None,
            oddsA=None,
            oddsB=None,
        )
        X = imputer.transform(row_df[feature_cols])
        p = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        st.error(f"Could not score this matchup: {e}")
        return

    pick = pa if p >= 0.5 else pb
    conf = p if p >= 0.5 else 1 - p
    bar_a = int(round(p * 100))
    bar_b = 100 - bar_a

    img_a = player_image_html(pa, size=80)
    img_b = player_image_html(pb, size=80)
    st.markdown(
        f"""
        <div class="ps-card">
          <div class="match-card" style="grid-template-columns: 1.4fr 1fr 1.4fr;">
            <div style="display:flex;gap:14px;align-items:center;">
              <div>{img_a}</div>
              <div>
                <div class="meta">{h(str(match_date))} · {h(str(surface))} {h(str(round_code) or '')}</div>
                <div class="name">{h(pa)}</div>
                <div class="meta">p(A wins) = {p*100:.1f}%</div>
                <div class="bar-bg"><div class="bar-fill bar-a" style="width:{bar_a}%;"></div></div>
              </div>
            </div>
            <div class="center">
              <div class="vs">PREDICTION</div>
              <div class="name" style="margin-top:6px;">{h(pick)}</div>
              <div class="meta">{confidence_label(p)} confidence · {conf*100:.1f}%</div>
            </div>
            <div style="display:flex;gap:14px;align-items:center;justify-content:flex-end;">
              <div style="text-align:right;">
                <div class="name">{h(pb)}</div>
                <div class="meta">p(B wins) = {(1-p)*100:.1f}%</div>
                <div class="bar-bg"><div class="bar-fill bar-b" style="width:{bar_b}%;"></div></div>
              </div>
              <div>{img_b}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 4])
    with cols[0]:
        st.button(f"Open profile: {pa}", key="w_open_pa", on_click=navigate_to_player, args=(pa,), width="stretch")
    with cols[1]:
        st.button(f"Open profile: {pb}", key="w_open_pb", on_click=navigate_to_player, args=(pb,), width="stretch")


# =============================================================================
# Tab: Leaderboard
# =============================================================================

def tab_leaderboard(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Leaderboard</div>", unsafe_allow_html=True)
    if history_df.empty:
        st.markdown('<div class="empty-state">No history dataset available.</div>', unsafe_allow_html=True)
        return

    surf_opts = sorted(history_df["surface"].dropna().unique().tolist())
    year_min = int(history_df["date"].dt.year.dropna().min())
    year_max = int(history_df["date"].dt.year.dropna().max())

    f1, f2, f3, f4 = st.columns([1.4, 2, 2, 1.2])
    with f1:
        min_matches = st.number_input("Min matches", min_value=10, max_value=1000, value=80, step=10, key="l_min")
    with f2:
        if year_min < year_max:
            year_range = st.slider(
                "Window",
                min_value=year_min,
                max_value=year_max,
                value=(max(year_min, year_max - 5), year_max),
                key="l_years",
            )
        else:
            year_range = (year_min, year_max)
    with f3:
        sel_surfaces = st.multiselect("Surface", surf_opts, default=surf_opts, key="l_surfaces")
    with f4:
        top_n = st.slider("Top N", 10, 200, 50, key="l_topn")

    df = history_df.copy()
    df = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])]
    if sel_surfaces:
        df = df[df["surface"].isin(sel_surfaces)]
    if df.empty:
        st.markdown('<div class="empty-state">No matches in the current selection.</div>', unsafe_allow_html=True)
        return

    wins = df["playerA"].value_counts()
    losses = df["playerB"].value_counts()
    lb = pd.DataFrame({"Wins": wins, "Losses": losses}).fillna(0)
    lb["Matches"] = lb["Wins"] + lb["Losses"]
    lb["Win rate"] = lb["Wins"] / lb["Matches"].replace(0, np.nan)
    lb = lb[lb["Matches"] >= int(min_matches)].sort_values(["Win rate", "Wins"], ascending=[False, False])
    lb = lb.head(int(top_n))
    lb.index.name = "Player"
    lb = lb.reset_index()
    lb.insert(0, "Rank", np.arange(1, len(lb) + 1))
    lb["Win rate"] = lb["Win rate"] * 100
    for c in ("Wins", "Losses", "Matches"):
        lb[c] = lb[c].astype(int)

    st.dataframe(
        lb[["Rank", "Player", "Matches", "Wins", "Losses", "Win rate"]],
        width="stretch",
        hide_index=True,
        height=560,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Wins": st.column_config.NumberColumn("Wins", width="small"),
            "Losses": st.column_config.NumberColumn("Losses", width="small"),
            "Matches": st.column_config.NumberColumn("Matches", width="small"),
            "Win rate": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
        },
    )

    # Jump-to-profile dropdown
    st.caption("Open a profile from the leaderboard:")
    pick = st.selectbox("Player", ["—"] + lb["Player"].tolist(), key="l_jump")
    if pick != "—":
        st.button(f"Open profile: {pick}", key="l_open", on_click=navigate_to_player, args=(pick,), width="stretch")


# =============================================================================
# Main
# =============================================================================

_init_state()
render_nav()
render_hero()

pred_df = load_predictions()
history_df = load_history()
render_kpis(pred_df)
render_top_nav_buttons()

view = st.session_state.get("view", "Matches")
if view == "Matches":
    tab_matches(pred_df)
elif view == "Upcoming":
    tab_upcoming(history_df)
elif view == "Players":
    tab_players(history_df)
elif view == "Tournaments":
    tab_tournaments(history_df)
elif view == "What-if":
    tab_whatif(history_df)
elif view == "Leaderboard":
    tab_leaderboard(history_df)
else:
    tab_matches(pred_df)
