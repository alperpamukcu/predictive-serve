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
from src.utils.config import DATA_DIR, MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.utils.country import flag_emoji
from src.utils.env import getenv, try_load_dotenv
from src.utils.feature_utils import load_feature_list
from src.utils.player_meta import (
    PlayerMeta,
    age_from_bday,
    attach_country,
    build_history_index,
    canonical_parts,
    find_player_in_fixtures,
    is_doubles,
    load_cache as load_player_cache,
    resolve_history_name,
    save_cache as save_player_cache,
)

try:
    from src.predict.whatif import build_feature_row  # type: ignore
except Exception:  # pragma: no cover
    build_feature_row = None  # type: ignore

try:
    from src.integrations.api_tennis import (  # type: ignore
        ApiTennisConfig,
        get_fixtures,
        get_h2h,
        get_livescore,
        get_players,
        get_standings,
    )
except Exception:  # pragma: no cover
    ApiTennisConfig = None  # type: ignore
    get_fixtures = None  # type: ignore
    get_h2h = None  # type: ignore
    get_livescore = None  # type: ignore
    get_players = None  # type: ignore
    get_standings = None  # type: ignore


# =============================================================================
# Page + assets
# =============================================================================

st.set_page_config(
    page_title="Predictive Serve",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


CSS = """
<style>
/* Twemoji web font for proper flag emoji rendering on Windows
   (Segoe UI Emoji shows letter pairs like 'IT' instead of 🇮🇹).
   Loaded from jsDelivr CDN; falls back to system emoji on slow networks. */
@font-face {
  font-family: "Twemoji Country Flags";
  unicode-range: U+1F1E6-1F1FF, U+1F3F4, U+E0062-E0063, U+E0065, U+E0067, U+E006C, U+E0073-E0074, U+E007F;
  src: url('https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1.8/dist/TwemojiCountryFlags.woff2') format('woff2');
  font-display: swap;
}
html, body, button, input, select, textarea, [class^="st-"], [class*=" st-"], .stMarkdown, .stApp {
  font-family: "Twemoji Country Flags", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
}

header[data-testid="stHeader"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
#MainMenu, footer { visibility: hidden; }

:root {
  --bg: #1c2333;
  --bg-2: #232b3d;
  --surface: rgba(255,255,255,0.05);
  --surface-2: rgba(255,255,255,0.085);
  --surface-strong: rgba(255,255,255,0.12);
  --line: rgba(255,255,255,0.10);
  --line-strong: rgba(255,255,255,0.20);
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
    radial-gradient(1200px 700px at 14% -10%, rgba(106,169,255,0.08), transparent 55%),
    radial-gradient(900px 540px at 90% -8%, rgba(156,135,255,0.06), transparent 55%),
    radial-gradient(800px 600px at 50% 115%, rgba(255,141,99,0.03), transparent 50%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
}

/* Scrollbars — slim, subtle, theme-aware */
* {
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.18) transparent;
}
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(106,169,255,0.32), rgba(156,135,255,0.32));
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(106,169,255,0.55), rgba(156,135,255,0.55));
  background-clip: padding-box;
  border: 2px solid transparent;
}
*::-webkit-scrollbar-corner { background: transparent; }

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
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
  letter-spacing: -0.005em !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover {
  border-color: var(--line-strong) !important;
  background: var(--surface-2) !important;
  transform: translateY(-1px);
}
/* Buttons that lead with 👤 are player chips — blue tint */
.stButton > button:has(> div > p:first-child[data-testid="stMarkdownContainer"]),
.stButton > button p:first-child {
  font-size: 0.92rem !important;
}
.stButton > button[data-testid] {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
/* Tag-style nav chips: target buttons by content via attribute hack — uses
   :has() which is supported in Chromium >=105. Also fall back to a
   generic visual treatment for everyone else. */
.stButton:has(button:has-text("👤")) > button { /* not all browsers */ }


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

/* Upcoming match card — mirrors the What-if visual language */
.up-card {
  padding: 16px 20px; margin-bottom: 12px;
  border-radius: var(--radius);
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}
.up-meta {
  color: var(--muted); font-size: 0.84rem; margin-bottom: 12px;
}
.up-row {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1.4fr;
  gap: 18px; align-items: center;
}
.up-side { display: flex; gap: 14px; align-items: center; }
.up-side-right { flex-direction: row-reverse; justify-content: flex-start; }
.up-side-right .up-info-right { text-align: right; }
.up-photo { flex: 0 0 auto; }
.up-photo > div, .up-photo > img {
  width: 72px !important; height: 72px !important; border-radius: 50%;
  border: 2px solid var(--line-strong); background: var(--surface);
}
.up-info { flex: 1 1 auto; min-width: 0; }
.up-name { color: #fff; font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; line-height: 1.15; }
.up-prob { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }
.up-prob b { color: var(--text); font-weight: 700; }
.up-pick {
  text-align: center; padding: 10px 12px;
  border-left: 1px dashed rgba(255,255,255,0.10);
  border-right: 1px dashed rgba(255,255,255,0.10);
}
.up-pick-label { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
.up-pick-name { color: #fff; font-weight: 800; font-size: 1.15rem; margin-top: 4px; letter-spacing: -0.01em; }
.up-pick-conf { color: var(--accent); font-size: 0.85rem; margin-top: 4px; font-weight: 600; }

/* Match row card (legacy, used by What-if) */
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

/* Live ticker — pinned just under the top nav, scrolls horizontally */
.live-ticker {
  display: flex; align-items: stretch; gap: 0;
  margin: 0 -1.6rem 12px -1.6rem;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255, 100, 113, 0.08), rgba(255, 100, 113, 0.02));
  overflow: hidden;
}
.live-ticker .ticker-label {
  flex: 0 0 auto;
  padding: 8px 16px;
  background: var(--bad);
  color: #fff;
  font-weight: 800; font-size: 0.78rem; letter-spacing: 0.12em;
  display: flex; align-items: center;
  position: relative;
  box-shadow: 0 0 24px rgba(255, 100, 113, 0.35);
}
.live-ticker.live-ticker-quiet {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
}
.live-ticker.live-ticker-quiet .ticker-label {
  background: var(--surface-strong);
  color: var(--muted);
  box-shadow: none;
}
.live-ticker .ticker-quiet-msg {
  padding: 8px 18px;
  color: var(--muted);
  font-size: 0.86rem;
  display: flex; align-items: center;
}
.live-ticker .ticker-track-wrap {
  flex: 1 1 auto; overflow: hidden; position: relative;
}
.live-ticker .ticker-track {
  display: inline-flex; gap: 36px; padding: 10px 18px; align-items: center;
  white-space: nowrap;
  animation: ps-ticker 180s linear infinite;
}
.live-ticker:hover .ticker-track { animation-play-state: paused; }
@keyframes ps-ticker {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}
.ticker-item {
  display: inline-flex; gap: 10px; align-items: center;
  font-size: 0.92rem; color: var(--text);
}
.ticker-item .ticker-photo > div, .ticker-item .ticker-photo > img {
  width: 36px !important; height: 36px !important; border-radius: 50%;
  border: 1px solid var(--line-strong);
  overflow: hidden;
}
.ticker-item .ticker-photo > div img,
.ticker-item .ticker-photo img { width: 36px !important; height: 36px !important; object-fit: cover; }
.ticker-item .ticker-name { font-weight: 700; color: #fff; }
.ticker-item .ticker-score {
  padding: 2px 10px; border-radius: 6px; font-weight: 800;
  background: rgba(255,255,255,0.06); color: #fff;
  border: 1px solid var(--line-strong);
}
.dot-live {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--bad);
  box-shadow: 0 0 10px var(--bad);
  animation: ps-pulse 1.4s ease-in-out infinite;
}
@keyframes ps-pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.85); }
  50% { opacity: 1; transform: scale(1.1); }
}

/* Leaderboard cards */
.lb-card {
  display: grid;
  grid-template-columns: 60px 80px minmax(180px, 1.4fr) repeat(5, minmax(72px, 1fr));
  align-items: center;
  gap: 12px;
  padding: 12px 18px;
  margin-bottom: 8px;
  border-radius: var(--radius);
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  transition: border-color 0.18s ease, transform 0.18s ease;
}
.lb-card:hover {
  border-color: rgba(106,169,255,0.28);
  transform: translateX(2px);
}
.lb-rank {
  font-size: 1.4rem; font-weight: 800;
  background: linear-gradient(180deg, #ffffff, #b8d6ff);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  letter-spacing: -0.02em; text-align: center;
}
.lb-photo > div, .lb-photo > img {
  width: 56px !important; height: 56px !important; border-radius: 50%;
  border: 2px solid var(--line-strong);
}
.lb-meta { min-width: 0; }
.lb-name {
  color: #fff; font-weight: 700; font-size: 1.03rem; letter-spacing: -0.01em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.lb-sub {
  color: var(--muted); font-size: 0.82rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.lb-stat { text-align: center; }
.lb-stat-label {
  font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--soft-muted); font-weight: 700;
}
.lb-stat-val {
  font-size: 1.05rem; font-weight: 800; color: #fff; margin-top: 2px;
  letter-spacing: -0.01em; line-height: 1.2;
}
.lb-stat-val.small { font-size: 0.85rem; font-weight: 700; }
.streak-good { color: var(--good) !important; }
.streak-bad { color: var(--bad) !important; }

/* LIVE / FINAL pills on individual upcoming cards */
.live-pill-card {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: var(--bad); color: #fff; font-size: 0.72rem; font-weight: 800;
  letter-spacing: 0.10em;
  margin-right: 8px;
  box-shadow: 0 0 14px rgba(255,100,113,0.45);
  animation: ps-pulse 1.6s ease-in-out infinite;
}
.fin-pill-card {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: var(--surface-2); color: var(--muted);
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.10em;
  margin-right: 8px; border: 1px solid var(--line-strong);
}
.up-time {
  font-weight: 700; color: var(--text);
  font-variant-numeric: tabular-nums;
}
.up-score {
  display: inline-block; padding: 3px 10px; border-radius: 6px;
  background: rgba(106,169,255,0.12); color: #fff;
  font-weight: 800; font-size: 0.85rem; margin-right: 8px;
  border: 1px solid rgba(106,169,255,0.25);
  font-variant-numeric: tabular-nums;
}
.up-actual {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: var(--good-soft); color: #afe9d3;
  font-weight: 600; font-size: 0.78rem;
  border: 1px solid rgba(45,210,154,0.30);
  margin-left: 4px;
}
.up-card-finished { opacity: 0.94; }
.up-card-finished .up-pick-name { color: var(--muted); }
.up-card-finished .up-pick-conf { color: var(--muted); }
.round-winner-photo > div, .round-winner-photo > img {
  border-color: var(--good) !important;
  box-shadow: 0 0 0 3px rgba(45,210,154,0.18) !important;
}

/* Top-right About button */
.ps-about-wrap { display:flex; align-items:center; height: 100%; padding-top: 14px; }
.ps-about-wrap .stButton > button {
  background: linear-gradient(180deg, rgba(106,169,255,0.18), rgba(106,169,255,0.06)) !important;
  border-color: rgba(106,169,255,0.40) !important;
  color: #fff !important;
  white-space: nowrap !important;
}

/* Coverage two-up grid */
.coverage-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 12px; margin-top: 8px;
}
@media (max-width: 880px) { .coverage-grid { grid-template-columns: 1fr; } }
.cov-card {
  padding: 14px 18px; border-radius: var(--radius);
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
}
.cov-card-good { border-left: 3px solid var(--good); }
.cov-card-bad  { border-left: 3px solid var(--bad); }
.cov-title {
  font-size: 0.92rem; font-weight: 700; color: #fff;
  margin-bottom: 8px; letter-spacing: -0.01em;
}
.cov-card ul { margin: 0; padding-left: 18px; color: var(--text); font-size: 0.88rem; line-height: 1.6; }
.cov-card ul i { color: var(--muted); font-style: italic; }

/* H2H card */
.h2h-card { padding: 16px 22px; }
.h2h-title { font-size: 0.92rem; font-weight: 700; color: #fff; margin-bottom: 12px; letter-spacing: -0.01em; }
.h2h-row { display: grid; grid-template-columns: 1.2fr 2fr 1.2fr; align-items: center; gap: 18px; }
.h2h-side { display: flex; gap: 10px; align-items: center; min-width: 0; }
.h2h-side-right { flex-direction: row-reverse; }
.h2h-photo > div, .h2h-photo > img {
  width: 48px !important; height: 48px !important; border-radius: 50%;
  border: 2px solid var(--line-strong);
}
.h2h-name {
  color: #fff; font-weight: 700; font-size: 0.92rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1 1 auto;
}
.h2h-side-right .h2h-name { text-align: right; }
.h2h-wins {
  font-size: 1.5rem; font-weight: 800; color: #fff;
  padding: 0 10px;
  font-variant-numeric: tabular-nums;
}
.h2h-bar {
  display: flex; height: 14px; border-radius: 7px; overflow: hidden;
  border: 1px solid var(--line);
}
.h2h-bar-a { background: linear-gradient(90deg, #ff7059, #ffb27a); }
.h2h-bar-b { background: linear-gradient(90deg, #6aa9ff, #9c87ff); }

/* Tournament round cards */
.round-card {
  padding: 12px 18px; margin-bottom: 8px;
  border-radius: 12px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  transition: border-color 0.18s ease, transform 0.18s ease;
}
.round-card:hover { border-color: rgba(106,169,255,0.30); transform: translateX(2px); }
.round-meta {
  color: var(--muted); font-size: 0.78rem; margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}
.round-row {
  display: grid; grid-template-columns: 1fr auto 1fr;
  align-items: center; gap: 16px;
}
.round-side { display: flex; gap: 12px; align-items: center; min-width: 0; }
.round-side:last-child { flex-direction: row-reverse; }
.round-side:last-child .round-name-wrap { text-align: right; align-items: flex-end; }
.round-photo > div, .round-photo > img {
  width: 44px !important; height: 44px !important; border-radius: 50%;
  border: 2px solid var(--line-strong);
}
.round-name-wrap { display: flex; flex-direction: column; min-width: 0; flex: 1 1 auto; }
.round-name {
  color: #fff; font-weight: 700; font-size: 0.98rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.round-flag { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
.round-side-winner .round-name { color: #fff; font-weight: 800; }
.round-side-winner .round-photo > div,
.round-side-winner .round-photo > img {
  border-color: var(--good);
  box-shadow: 0 0 0 3px rgba(45,210,154,0.15);
}
.round-badge-w {
  display: inline-block; padding: 2px 8px; margin-left: 6px;
  border-radius: 999px; font-size: 0.62rem; letter-spacing: 0.10em;
  background: var(--good-soft); color: #afe9d3; font-weight: 800;
  vertical-align: middle;
}
.round-score {
  padding: 8px 16px; border-radius: 10px;
  background: rgba(106,169,255,0.10);
  border: 1px solid rgba(106,169,255,0.28);
  color: #fff; font-weight: 800; font-variant-numeric: tabular-nums;
  font-size: 1.0rem; letter-spacing: 0.04em;
  min-width: 90px; text-align: center;
}

/* Tournament sub-header inside Upcoming */
.tour-subheader {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 8px 14px; margin: 4px 0 6px 0;
  border-radius: 8px;
  background: rgba(106,169,255,0.06);
  border-left: 3px solid var(--accent);
  color: #fff;
  font-weight: 700; font-size: 0.95rem; letter-spacing: -0.01em;
}
.tour-subheader .count {
  color: var(--muted); font-weight: 500; font-size: 0.82rem;
}

/* Player profile header */
.profile-header { display:flex; gap:20px; align-items:center; padding:20px 22px;
  border:1px solid var(--line); border-radius: var(--radius-lg);
  background:
    radial-gradient(600px 220px at 0% 0%, rgba(106,169,255,0.10), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  margin-bottom: 14px;
}
.profile-header .avatar { flex: 0 0 auto; width:120px; height:120px; }
.profile-header .meta-block { flex: 1 1 auto; min-width: 0; }
.profile-header .meta-block .name { font-size:1.85rem; font-weight:800; color:#fff; letter-spacing:-0.02em; line-height: 1.1; }
.profile-header .meta-block .name .flag { font-size:1.6rem; margin-right:10px; }
.profile-header .meta-block .sub { color:var(--muted); margin-top:6px; font-size:0.95rem; }
.profile-header .meta-chips { margin-top: 10px; display:flex; flex-wrap:wrap; gap:8px; }
.profile-header .chip {
  padding:5px 12px; border-radius:999px;
  background: var(--surface-2); border:1px solid var(--line);
  color: var(--text); font-size:0.82rem; font-weight:600;
}

/* Tournament hero card */
.tour-hero {
  padding: 22px 24px; margin-bottom: 14px;
  border-radius: var(--radius-lg); border:1px solid var(--line);
  background:
    radial-gradient(700px 200px at 90% -20%, rgba(255,141,99,0.12), transparent 55%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  display:flex; gap:18px; align-items:center; justify-content: space-between;
}
.tour-hero .name { font-size: 1.7rem; font-weight: 800; color:#fff; letter-spacing:-0.02em; line-height: 1.1; }
.tour-hero .sub { color: var(--muted); margin-top: 6px; }
.tour-hero .badges { display:flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.tour-hero .badge {
  padding: 5px 11px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
  background: var(--surface-2); border: 1px solid var(--line); color: var(--text);
}

/* Details disclosure */
details.ps-details {
  margin: -4px 0 12px 0;
}
details.ps-details summary {
  cursor: pointer; color: var(--muted); font-size: 0.85rem;
  letter-spacing: 0.04em; text-transform: uppercase; font-weight: 700;
  padding: 8px 0; user-select: none;
}
details.ps-details summary::-webkit-details-marker { color: var(--muted); }
details.ps-details[open] summary { color: var(--text); }
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


def _canonical_key(name: str) -> str:
    """Stable string key for a player (combines initial + surname)."""
    i, sur = canonical_parts(str(name))
    return f"{i or ''}|{sur}" if sur else ""


@st.cache_data(show_spinner=False)
def player_directory(history_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player career stats, **deduplicated by canonical key**
    so 'Carreno Busta P.' and 'Carreno-Busta P.' merge into one row."""
    if history_df.empty:
        return pd.DataFrame(columns=["player", "matches", "wins", "losses", "first_year", "last_year", "winrate"])

    a = history_df.groupby("playerA").agg(wins=("playerA", "size"), first_a=("date", "min"), last_a=("date", "max"))
    b = history_df.groupby("playerB").agg(losses=("playerB", "size"), first_b=("date", "min"), last_b=("date", "max"))
    out = pd.concat([a, b], axis=1).fillna({"wins": 0, "losses": 0})
    out[["wins", "losses"]] = out[["wins", "losses"]].astype(int)
    first = pd.concat([out["first_a"], out["first_b"]], axis=1).min(axis=1)
    last = pd.concat([out["last_a"], out["last_b"]], axis=1).max(axis=1)
    out = out.assign(first=first, last=last)
    out = out.drop(columns=[c for c in ["first_a", "first_b", "last_a", "last_b"] if c in out.columns])
    out = out.reset_index().rename(columns={"index": "player"})
    if "playerA" in out.columns:
        out = out.rename(columns={"playerA": "player"})

    # Dedup by canonical (initial, surname). The display name keeps the
    # spelling that occurs in the largest number of matches.
    out["__key"] = out["player"].apply(_canonical_key)
    out = out[out["__key"] != ""].copy()
    out["__matches_for_pick"] = out["wins"] + out["losses"]

    grouped = (
        out.sort_values("__matches_for_pick", ascending=False)
        .groupby("__key", as_index=False)
        .agg(
            player=("player", "first"),  # most-frequent spelling, sort_values above
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            first=("first", "min"),
            last=("last", "max"),
        )
    )
    grouped["matches"] = grouped["wins"] + grouped["losses"]
    grouped["winrate"] = grouped["wins"] / grouped["matches"].replace(0, np.nan)
    grouped["first_year"] = pd.to_datetime(grouped["first"], errors="coerce").dt.year
    grouped["last_year"] = pd.to_datetime(grouped["last"], errors="coerce").dt.year
    grouped = grouped.drop(columns=["first", "last"])
    grouped = grouped.sort_values(["matches", "wins"], ascending=[False, False])
    return grouped[["player", "matches", "wins", "losses", "first_year", "last_year", "winrate"]]


@st.cache_data(show_spinner=False)
def _name_variants_index(history_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map canonical key -> every history spelling that maps to it."""
    if history_df.empty:
        return {}
    names = pd.concat([history_df["playerA"], history_df["playerB"]]).dropna().astype(str).unique()
    out: Dict[str, List[str]] = {}
    for n in names:
        k = _canonical_key(n)
        if not k:
            continue
        out.setdefault(k, []).append(n)
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
    """
    Pull a moderately wide window of fixtures for player metadata lookup.
    The API rejects very wide windows (500), so we chunk into 14-day slices
    that we know work and merge the results.
    """
    if get_fixtures is None or ApiTennisConfig is None:
        return []
    key = _api_key()
    if not key:
        return []
    cfg = ApiTennisConfig(api_key=key, cache_ttl_s=86400)
    today = dt.date.today()
    chunks = [
        (today - dt.timedelta(days=14), today),
        (today, today + dt.timedelta(days=14)),
    ]
    out: List[Dict[str, Any]] = []
    for start, stop in chunks:
        try:
            out.extend(get_fixtures(cfg, start, stop))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Player metadata: nationality, age, photo (via API-Tennis get_players)
# ---------------------------------------------------------------------------

PLAYER_META_DIR = DATA_DIR / "cache"


def _load_player_cache() -> dict:
    return load_player_cache(PLAYER_META_DIR)


def _save_player_cache(cache: dict) -> None:
    save_player_cache(PLAYER_META_DIR, cache)


def _download_photo(name: str, logo_url: Optional[str]) -> bool:
    if not logo_url or not logo_url.startswith("http"):
        return False
    out = ASSETS_DIR / "players" / f"{slugify(name)}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(logo_url, timeout=20)
        r.raise_for_status()
        out.write_bytes(r.content)
        return True
    except Exception:
        return False


def get_player_meta(name: str) -> PlayerMeta:
    """
    Idempotent metadata lookup with **enrichment-first** semantics:
      1) Returns the cached entry if present (never overwrites known fields).
      2) For new names, tries fixtures (gets player_key + photo URL in one go).
      3) For cached entries with a key but no local photo, downloads the photo
         lazily on first read.
    A retired player whose entry has only ``full_name + country`` keeps that
    metadata even if a later live lookup can't find their key.
    """
    cache = _load_player_cache()
    meta = cache.get(name)

    # 1) Already in cache (and resolved). Return as-is, but lazily download
    #    a missing photo. not_found entries from earlier (broken) lookups
    #    fall through to a fresh resolution below.
    if meta is not None and not meta.not_found:
        if meta.player_key and not find_image(ASSETS_DIR / "players" / slugify(name)):
            if meta.logo_url:
                _download_photo(name, meta.logo_url)
            elif ApiTennisConfig is not None and get_players is not None and _api_key():
                try:
                    cfg = ApiTennisConfig(api_key=_api_key(), cache_ttl_s=86400)
                    rec = get_players(cfg, meta.player_key)
                    logo = rec.get("player_logo")
                    bday = rec.get("player_bday")
                    if logo and not meta.logo_url:
                        meta.logo_url = logo
                    if bday and not meta.birthday:
                        meta.birthday = bday
                        meta.age = age_from_bday(bday)
                    cache[name] = meta
                    _save_player_cache(cache)
                    if logo:
                        _download_photo(name, logo)
                except Exception:
                    pass
        return meta

    # 2) Brand new lookup.
    meta = PlayerMeta(name=name)

    if ApiTennisConfig is None or get_players is None or _api_key() is None:
        meta.not_found = True
        meta.fetched_at = dt.datetime.utcnow().isoformat() + "Z"
        cache[name] = attach_country(meta)
        _save_player_cache(cache)
        return cache[name]

    fixtures = cached_recent_api_fixtures()
    player_key, api_name, logo_url = find_player_in_fixtures(name, fixtures)
    meta.player_key = player_key
    meta.logo_url = logo_url

    if player_key:
        try:
            cfg = ApiTennisConfig(api_key=_api_key(), cache_ttl_s=86400)
            record = get_players(cfg, player_key)
            meta.full_name = record.get("player_full_name") or record.get("player_name") or api_name
            meta.country = record.get("player_country")
            meta.birthday = record.get("player_bday")
            meta.age = age_from_bday(meta.birthday)
            meta.logo_url = record.get("player_logo") or meta.logo_url
        except Exception:
            pass

    meta = attach_country(meta)
    meta.not_found = meta.player_key is None and not meta.country and not meta.logo_url
    meta.fetched_at = dt.datetime.utcnow().isoformat() + "Z"
    cache[name] = meta
    _save_player_cache(cache)

    if meta.logo_url:
        _download_photo(name, meta.logo_url)
    return meta


@st.cache_data(show_spinner=False, ttl=120)
def _canonical_to_history_index() -> Dict[Tuple[Optional[str], str], str]:
    """Index canonical key -> the history-style spelling we already have
    metadata for, so live-API names ('R. Bautista-Agut') can be mapped
    onto the slug we used to cache the photo ('bautista-agut-r')."""
    cache = _load_player_cache()
    out: Dict[Tuple[Optional[str], str], str] = {}
    for cached_name in cache:
        key = canonical_parts(cached_name)
        if key[1]:
            out.setdefault(key, cached_name)
    return out


def resolve_to_history_name(name: str) -> str:
    """Map any incoming name (history or API format) to the history-format
    we cached metadata under. Falls back to the input when unknown."""
    cache = _load_player_cache()
    if name in cache:
        return name
    key = canonical_parts(name)
    if not key[1]:
        return name
    return _canonical_to_history_index().get(key, name)


def display_name(name: str) -> str:
    """Use the API-Tennis full_name (e.g. ``Jannik Sinner``) when cached
    under any equivalent spelling, falling back to the input."""
    history_name = resolve_to_history_name(name)
    cache = _load_player_cache()
    meta = cache.get(history_name)
    if meta and meta.full_name:
        return meta.full_name
    return name


def player_label(name: str) -> str:
    """Dropdown label — flag + full name when metadata is cached."""
    cache = _load_player_cache()
    meta = cache.get(name)
    flag = meta.flag if (meta and meta.flag and meta.flag != "🏳️") else ""
    full = (meta.full_name if (meta and meta.full_name) else name)
    return f"{flag}  {full}".strip()


# ---------------------------------------------------------------------------
# Bulk metadata prefetch — ATP standings + parallel get_players for photos
# ---------------------------------------------------------------------------

def _build_standings_index() -> Dict[Tuple[Optional[str], str], Dict[str, Any]]:
    """Pull ATP standings (2k+ players) and index by (initial, surname)."""
    if get_standings is None or ApiTennisConfig is None:
        return {}
    key = _api_key()
    if not key:
        return {}
    cfg = ApiTennisConfig(api_key=key, cache_ttl_s=86400)
    try:
        rows = get_standings(cfg, "ATP")
    except Exception:
        return {}
    index: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    surname_only: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        full = (r.get("player") or "").strip()
        if not full:
            continue
        ci = canonical_parts(full)
        if not ci[1]:
            continue
        index.setdefault(ci, r)
        surname_only.setdefault(ci[1], r)
    # Merge surname-only fallbacks under (None, surname) key
    for sur, row in surname_only.items():
        index.setdefault((None, sur), row)
    return index


def prefetch_via_standings(history_names: List[str], progress_cb=None) -> Tuple[int, int]:
    """Resolve as many history names as possible via a single ATP-standings
    call (one API request gets us 2k+ players' full names + countries +
    player_keys), then download photos in parallel.

    Returns ``(resolved_count, total_attempted)``.
    """
    import concurrent.futures

    cache = _load_player_cache()
    standings_idx = _build_standings_index()
    if not standings_idx:
        return 0, 0

    # First pass: attach full_name + country + player_key from standings.
    todo_for_photo: list[Tuple[str, int]] = []
    new_resolved = 0
    for name in history_names:
        if not name:
            continue
        if name in cache and cache[name].fetched_at and not cache[name].not_found:
            continue
        i, sur = canonical_parts(name)
        if not sur:
            continue
        row = (
            standings_idx.get((i, sur))
            or standings_idx.get((None, sur))
        )
        if not row:
            continue
        meta = PlayerMeta(name=name)
        meta.full_name = (row.get("player") or "").strip() or None
        meta.country = (row.get("country") or "").strip() or None
        try:
            meta.player_key = int(row.get("player_key")) if row.get("player_key") else None
        except Exception:
            meta.player_key = None
        meta = attach_country(meta)
        meta.fetched_at = dt.datetime.utcnow().isoformat() + "Z"
        meta.not_found = meta.player_key is None
        cache[name] = meta
        new_resolved += 1
        if meta.player_key:
            todo_for_photo.append((name, meta.player_key))

    _save_player_cache(cache)

    # Second pass: parallel get_players for photo + birthday for the players
    # we haven't yet downloaded a photo for.
    def _fetch_photo(item: Tuple[str, int]) -> bool:
        name, player_key = item
        if find_image(ASSETS_DIR / "players" / slugify(name)):
            return True  # already have a photo
        try:
            cfg = ApiTennisConfig(api_key=_api_key(), cache_ttl_s=86400)
            record = get_players(cfg, player_key)
            logo_url = record.get("player_logo") or ""
            bday = record.get("player_bday") or ""
            local_cache = _load_player_cache()
            if name in local_cache:
                if logo_url:
                    local_cache[name].logo_url = logo_url
                if bday:
                    local_cache[name].birthday = bday
                    local_cache[name].age = age_from_bday(bday)
                _save_player_cache(local_cache)
            if logo_url and logo_url.startswith("http"):
                out = ASSETS_DIR / "players" / f"{slugify(name)}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(logo_url, timeout=20)
                r.raise_for_status()
                out.write_bytes(r.content)
                return True
        except Exception:
            return False
        return False

    if todo_for_photo:
        total = len(todo_for_photo)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_photo, item): item for item in todo_for_photo}
            for i, _fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                if progress_cb:
                    progress_cb(i, total)

    return new_resolved, len(history_names)


def fmt_date_long(d) -> str:
    """``2026-04-28`` -> ``28 April 2026``. Cross-platform (no %-d / %#d)."""
    if d is None or pd.isna(d):
        return ""
    ts = pd.Timestamp(d)
    return f"{ts.day} {ts.strftime('%B')} {ts.year}"


# ---------------------------------------------------------------------------
# Live scores
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60)
def cached_livescore() -> List[Dict[str, Any]]:
    """Pull api-tennis.com /get_livescore. Cached for 60s so the ticker can
    refresh without spamming the provider."""
    if get_livescore is None or ApiTennisConfig is None:
        return []
    key = _api_key()
    if not key:
        return []
    cfg = ApiTennisConfig(api_key=key, cache_ttl_s=60)
    try:
        return list(get_livescore(cfg))
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def cached_h2h(player_a_key: int, player_b_key: int) -> Dict[str, Any]:
    if get_h2h is None or ApiTennisConfig is None:
        return {}
    key = _api_key()
    if not key:
        return {}
    cfg = ApiTennisConfig(api_key=key, cache_ttl_s=86400)
    try:
        return get_h2h(cfg, player_a_key, player_b_key)
    except Exception:
        return {}


def render_h2h(player_a: str, player_b: str) -> None:
    """Render a compact head-to-head panel if both players have API keys
    in the metadata cache."""
    meta_a = get_player_meta(player_a)
    meta_b = get_player_meta(player_b)
    if not (meta_a and meta_a.player_key and meta_b and meta_b.player_key):
        return
    payload = cached_h2h(int(meta_a.player_key), int(meta_b.player_key))
    if not payload:
        return
    matches = payload.get("H2H") or []
    if not isinstance(matches, list) or not matches:
        return

    # Count wins per player_key
    wins_a = wins_b = 0
    for m in matches:
        winner = (m.get("event_winner") or "").lower()
        first = m.get("event_first_player")
        # API picks "First Player" or "Second Player"; we know each row's
        # first_player_key, so map it back to our local A/B.
        first_key = m.get("first_player_key")
        if first_key is None:
            continue
        is_first_a = int(first_key) == int(meta_a.player_key)
        if "first" in winner:
            if is_first_a:
                wins_a += 1
            else:
                wins_b += 1
        elif "second" in winner:
            if is_first_a:
                wins_b += 1
            else:
                wins_a += 1

    total = wins_a + wins_b
    if total == 0:
        return
    pct_a = wins_a / total * 100
    pct_b = 100 - pct_a

    name_a = display_name(player_a)
    name_b = display_name(player_b)
    img_a = player_image_html(player_a, size=48)
    img_b = player_image_html(player_b, size=48)

    st.markdown(
        f"""
        <div class="ps-card h2h-card">
          <div class="h2h-title">Head-to-Head &middot; <span style="color:var(--muted);font-weight:500;">{total} previous meeting{'s' if total != 1 else ''}</span></div>
          <div class="h2h-row">
            <div class="h2h-side">
              <div class="h2h-photo">{img_a}</div>
              <div class="h2h-name">{h(name_a)}</div>
              <div class="h2h-wins">{wins_a}</div>
            </div>
            <div class="h2h-bar">
              <div class="h2h-bar-a" style="width:{pct_a:.1f}%;"></div>
              <div class="h2h-bar-b" style="width:{pct_b:.1f}%;"></div>
            </div>
            <div class="h2h-side h2h-side-right">
              <div class="h2h-wins">{wins_b}</div>
              <div class="h2h-name">{h(name_b)}</div>
              <div class="h2h-photo">{img_b}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _live_for_player(player: str, livescores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    target = canonical_parts(player)
    for ev in livescores:
        for nf in ("event_first_player", "event_second_player"):
            api_name = (ev.get(nf) or "").strip()
            if not api_name or is_doubles(api_name):
                continue
            if canonical_parts(api_name) == target:
                return ev
    return None


def render_live_ticker() -> None:
    """A thin newscaster-style ticker pinned just under the top nav. Renders
    even when nothing is live so users can tell the feed is wired up."""
    events = cached_livescore()
    singles = [
        ev for ev in events
        if not is_doubles((ev.get("event_first_player") or ""))
        and not is_doubles((ev.get("event_second_player") or ""))
    ]

    if not singles:
        st.markdown(
            """
            <div class="live-ticker live-ticker-quiet">
              <div class="ticker-label">LIVE</div>
              <div class="ticker-quiet-msg">No matches in progress right now &middot; check Upcoming for the next schedule</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    items_html: list[str] = []
    for ev in singles[:24]:
        a = (ev.get("event_first_player") or "").strip()
        b = (ev.get("event_second_player") or "").strip()
        score = (ev.get("event_game_result") or ev.get("event_final_result") or "").strip()
        if not a or not b:
            continue
        a_disp = display_name(a)
        b_disp = display_name(b)
        a_img = ticker_photo_html(a, size=36)
        b_img = ticker_photo_html(b, size=36)
        items_html.append(
            f'<div class="ticker-item">'
            f'<span class="dot-live"></span>'
            f'{a_img}'
            f'<span class="ticker-name">{h(a_disp)}</span>'
            f'<span class="ticker-score">{h(score) or "-"}</span>'
            f'<span class="ticker-name">{h(b_disp)}</span>'
            f'{b_img}'
            f'</div>'
        )
    if not items_html:
        return
    items_html_doubled = "".join(items_html * 2)  # duplicate so the loop is seamless
    st.markdown(
        f"""
        <div class="live-ticker">
          <div class="ticker-label">LIVE &middot; {len(singles)}</div>
          <div class="ticker-track-wrap"><div class="ticker-track">{items_html_doubled}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_upcoming_card(r: pd.Series, uid: str, *, state: str = "scheduled") -> None:
    """Render a single What-if-style match card with live/finished variants.

    state = 'live'      -> red LIVE pill + current game score on top
            'scheduled' -> AI pick + probability bars
            'finished'  -> dim card + final score + ✓ on actual winner side
    """
    pa = str(r["playerA"])
    pb = str(r["playerB"])
    pa_disp = display_name(pa)
    pb_disp = display_name(pb)
    p_a = float(r["p_model"])
    p_b = 1.0 - p_a
    bar_a = int(round(p_a * 100))
    bar_b = 100 - bar_a
    img_a = player_image_html(pa, size=72)
    img_b = player_image_html(pb, size=72)
    time_str = (r.get("event_time") or "").strip()
    surface = str(r.get("surface") or "")
    round_str = str(r.get("round") or "")

    meta_pieces = []
    if time_str:
        meta_pieces.append(f"<span class='up-time'>🕐 {h(time_str)}</span>")
    if round_str:
        meta_pieces.append(h(round_str))
    if surface:
        meta_pieces.append(h(surface))
    meta_html = " · ".join(meta_pieces)

    badge_html = ""
    extra_class = ""
    if state == "live":
        score = (r.get("score") or r.get("live_game") or "").strip()
        badge_html = f"<span class='live-pill-card'>LIVE</span>"
        if score:
            badge_html += f"<span class='up-score'>{h(score)}</span>"
    elif state == "finished":
        extra_class = "up-card-finished"
        score = (r.get("score") or "").strip()
        winner_side = (r.get("winner_side") or "").lower()
        actual_winner_disp = pa_disp if "first" in winner_side else pb_disp if "second" in winner_side else None
        badge_html = "<span class='fin-pill-card'>FINAL</span>"
        if score:
            badge_html += f"<span class='up-score'>{h(score)}</span>"
        if actual_winner_disp:
            badge_html += f"<span class='up-actual'>Winner · <b>{h(actual_winner_disp)}</b></span>"

    pick_disp = display_name(str(r["winner_pick"]))
    pick_label = "AI PICK" if state != "finished" else "AI PICKED"
    pick_block = (
        f"<div class='up-pick'>"
        f"<div class='up-pick-label'>{pick_label}</div>"
        f"<div class='up-pick-name'>{h(pick_disp)}</div>"
        f"<div class='up-pick-conf'>{confidence_label(float(r['winner_prob']))} · "
        f"{r['winner_prob']*100:.1f}%</div>"
        f"</div>"
    )

    chance_a_label = f"{h(pa_disp.split()[-1] if pa_disp else 'A')}'s chance to win"
    chance_b_label = f"{h(pb_disp.split()[-1] if pb_disp else 'B')}'s chance to win"

    st.markdown(
        f"""
        <div class="up-card {extra_class}">
          <div class="up-meta">{badge_html}{meta_html}</div>
          <div class="up-row">
            <div class="up-side">
              <div class="up-photo">{img_a}</div>
              <div class="up-info">
                <div class="up-name">{h(pa_disp)}</div>
                <div class="up-prob">{chance_a_label}: <b>{p_a*100:.1f}%</b></div>
                <div class="bar-bg"><div class="bar-fill bar-a" style="width:{bar_a}%;"></div></div>
              </div>
            </div>
            {pick_block}
            <div class="up-side up-side-right">
              <div class="up-info up-info-right">
                <div class="up-name">{h(pb_disp)}</div>
                <div class="up-prob">{chance_b_label}: <b>{p_b*100:.1f}%</b></div>
                <div class="bar-bg"><div class="bar-fill bar-b" style="width:{bar_b}%;"></div></div>
              </div>
              <div class="up-photo">{img_b}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_action_buttons(
        prefix=uid,
        player_a=pa,
        player_b=pb,
        tournament=str(r.get("tournament") or "") or None,
    )


def _render_upcoming_for_player(player: str, history_df: pd.DataFrame) -> None:
    """If *player* has any fixture in fixtures_upcoming.csv (resolved against
    our history names), show a prominent banner with the AI prediction."""
    fix = load_real_fixtures()
    if fix.empty or "playerA" not in fix.columns or "playerB" not in fix.columns:
        return
    by_init_sur, by_surname, _ = _history_player_index(history_df)
    target = _canonical_key(player)
    if not target:
        return

    fix = fix.copy()
    fix["date"] = pd.to_datetime(fix["date"], errors="coerce")
    fix = fix.dropna(subset=["date", "playerA", "playerB"])
    fix = fix[fix["date"] >= pd.Timestamp.today().normalize()]
    if fix.empty:
        return
    fix = fix[~fix["playerA"].astype(str).map(is_doubles)]
    fix = fix[~fix["playerB"].astype(str).map(is_doubles)]
    fix["playerA_resolved"] = fix["playerA"].apply(lambda n: resolve_history_name(str(n), by_init_sur, by_surname))
    fix["playerB_resolved"] = fix["playerB"].apply(lambda n: resolve_history_name(str(n), by_init_sur, by_surname))
    mask = (
        fix["playerA_resolved"].apply(lambda n: _canonical_key(n or "") == target)
        | fix["playerB_resolved"].apply(lambda n: _canonical_key(n or "") == target)
    )
    rows = fix[mask].sort_values("date").head(2)
    if rows.empty:
        return

    sc = _score_fixtures(
        rows.assign(
            playerA=rows["playerA_resolved"],
            playerB=rows["playerB_resolved"],
        ),
        history_df,
    )
    if sc.empty:
        return

    st.markdown("<div class='ps-section-title'>Next match</div>", unsafe_allow_html=True)
    for _, r in sc.iterrows():
        meta_bits = []
        if r.get("tournament"):
            meta_bits.append(str(r["tournament"]))
        if r.get("surface"):
            meta_bits.append(str(r["surface"]))
        if r.get("round"):
            meta_bits.append(str(r["round"]))
        meta = " · ".join(meta_bits)
        pa = str(r["playerA"])
        pb = str(r["playerB"])
        pa_disp = display_name(pa)
        pb_disp = display_name(pb)
        pick_disp = display_name(str(r["winner_pick"]))
        p_a = float(r["p_model"])
        p_b = 1.0 - p_a
        bar_a = int(round(p_a * 100))
        bar_b = 100 - bar_a
        st.markdown(
            f"""
            <div class="up-card" style="border-color: rgba(106,169,255,0.35);">
              <div class="up-meta">{h(fmt_date_long(r['date']))} &middot; {h(meta)}</div>
              <div class="up-row">
                <div class="up-side">
                  <div class="up-photo">{player_image_html(pa, size=72)}</div>
                  <div class="up-info">
                    <div class="up-name">{h(pa_disp)}</div>
                    <div class="up-prob">p(A wins) = <b>{p_a*100:.1f}%</b></div>
                    <div class="bar-bg"><div class="bar-fill bar-a" style="width:{bar_a}%;"></div></div>
                  </div>
                </div>
                <div class="up-pick">
                  <div class="up-pick-label">AI PICK</div>
                  <div class="up-pick-name">{h(pick_disp)}</div>
                  <div class="up-pick-conf">{confidence_label(float(r['winner_prob']))} · {r['winner_prob']*100:.1f}%</div>
                </div>
                <div class="up-side up-side-right">
                  <div class="up-info up-info-right">
                    <div class="up-name">{h(pb_disp)}</div>
                    <div class="up-prob">p(B wins) = <b>{p_b*100:.1f}%</b></div>
                    <div class="bar-bg"><div class="bar-fill bar-b" style="width:{bar_b}%;"></div></div>
                  </div>
                  <div class="up-photo">{player_image_html(pb, size=72)}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def ticker_photo_html(name: str, size: int = 36) -> str:
    """Compact photo tile for the live ticker.

    Uses an inline <img> with explicit box-sizing so the ticker's small
    photos never end up squashed even when the underlying file has an
    unusual aspect ratio."""
    history_name = resolve_to_history_name(name)
    p = ASSETS_DIR / "players" / slugify(history_name)
    img = find_image(p)
    src = None
    if img and img.exists():
        try:
            data = base64.b64encode(img.read_bytes()).decode()
            ext = img.suffix.lower().lstrip(".")
            mime = "jpeg" if ext == "jpg" else ext
            src = f"data:image/{mime};base64,{data}"
        except Exception:
            src = None
    if src is None:
        src = svg_avatar_data_uri(name, size)
    return (
        f'<img src="{src}" alt="" '
        f'style="width:{size}px;height:{size}px;border-radius:50%;'
        f'border:1px solid rgba(255,255,255,0.18);box-sizing:border-box;'
        f'object-fit:cover;display:block;"/>'
    )


def player_image_html(name: str, size: int = 120) -> str:
    """Return an inline <img> tag — local cache first, otherwise SVG avatar."""
    history_name = resolve_to_history_name(name)
    p = ASSETS_DIR / "players" / slugify(history_name)
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


def _render_about_dialog() -> None:
    """Decorator-style dialog that fires when the top-right button is clicked.
    Uses st.dialog when available (1.31+); falls back to an expander."""
    info = load_metrics_json()
    model_label = str(info.get("model", "Model"))
    val = info.get("validation") or {}
    test = info.get("test") or {}
    blend = info.get("blend") or {}
    pred_df = load_predictions()
    n_matches = len(pred_df) if not pred_df.empty else 0
    earliest = pd.to_datetime(pred_df["date"]).min().date() if not pred_df.empty else None
    latest = pd.to_datetime(pred_df["date"]).max().date() if not pred_df.empty else None
    api_supp = PROCESSED_DIR / "recent_results_apitennis.csv"
    api_n = 0
    if api_supp.exists():
        try:
            api_n = len(pd.read_csv(api_supp))
        except Exception:
            api_n = 0

    val_acc = val.get("accuracy")
    val_ll = val.get("logloss")
    test_acc = test.get("accuracy")
    blend_acc = (blend.get("val") or {}).get("blend", {}).get("accuracy")
    alpha = blend.get("alpha")

    def _md(value, n=2):
        return "—" if value is None else f"{value*100:.{n}f}%"

    st.markdown(
        f"""
        ### About Predictive Serve

        **Predictive Serve** is a transparent, end-to-end tennis match
        forecasting console. Everything below is wired up in this app —
        nothing is mocked.

        #### What you're looking at
        - **Matches** — every historical match scored with the AI's
          probability + a green/red row tint showing whether the AI was
          right.
        - **Upcoming** — live fixtures grouped by tournament with
          probability bars, **LIVE** badges, and **FINAL** cards when
          today's matches end.
        - **Live ticker** — a strip at the top of the page scrolling the
          current set scores of every match in progress.
        - **Players / Tournaments** — per-entity profiles with photos,
          flags, full names, Elo trajectory and round breakdowns.
        - **What-if** — pick any two players + surface + date and the
          model returns the probability split, with a **head-to-head**
          card underneath when both players have API metadata.
        - **Leaderboard** — ranked by Win rate / Last 30d / AI accuracy /
          Streak, with country flag + full name + photo.

        #### Data sources
        - **tennis-data.co.uk** — canonical historical archive
          (2000 → today), refreshed nightly. Currently
          **{n_matches:,} matches** from **{earliest}** to **{latest}**.
        - **api-tennis.com** — last 21 days of finished singles matches
          merged into the history each night
          (currently **{api_n}** supplemented matches). Same endpoint
          also drives the live ticker, upcoming fixtures, player photos,
          country flags and ATP rankings.

        #### Model
        - **Active**: **{h(model_label)}** — a calibrated
          gradient-boosted tree (HistGradientBoosting / LightGBM
          candidates evaluated against a held-out 2025 test split).
        - **Features (40 total)**: global + surface Elo, last-5/10-match
          form, head-to-head **including surface-specific H2H**,
          rest days, workload, rank, round importance, tournament tier,
          set-level win rate. Bookmaker odds are **never** features
          ("edge" stays meaningful).
        - **Market-prior blend**: at inference time we mix
          α·p\\_model + (1-α)·p\\_market when an odds line exists.
          Current α from validation: **{0 if alpha is None else alpha:.2f}**
          (lower means "trust the market more on matches with lines").
        - **Metrics today**:
          - Pure AI val accuracy: **{_md(val_acc)}** (log loss
            **{(val_ll if val_ll is not None else 0):.3f}**)
          - Pure AI 2025 test accuracy: **{_md(test_acc)}**
          - Market-blend val accuracy (where odds exist):
            **{_md(blend_acc)}**

        #### Engineering
        - All artifacts (predictions CSV, model PKL, player metadata,
          photos) are regenerated by a **daily GitHub Actions cron** at
          04:00 UTC and committed to the repo, so the deployed app is
          always fresh without manual steps.
        - **No data leakage by construction**: feature_utils.py owns
          the single `LEAKY_MARKET_COLS` allowlist and `select_model_features`
          enforces it in every training and scoring path.
        """,
        unsafe_allow_html=False,
    )


def render_nav() -> None:
    info = load_metrics_json()
    model_label = str(info.get("model", "Model"))
    val = info.get("validation") or {}
    pill = "Live model"
    if isinstance(val.get("logloss"), (int, float)) and isinstance(val.get("accuracy"), (int, float)):
        pill = f"{model_label} | acc {val['accuracy']*100:.1f}% | log loss {val['logloss']:.3f}"

    nav_l, nav_r = st.columns([8, 1])
    with nav_l:
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
    with nav_r:
        st.markdown('<div class="ps-about-wrap">', unsafe_allow_html=True)
        if st.button("ℹ️ About", key="about_btn", width="stretch", help="Data sources, model and engineering details"):
            st.session_state["_show_about"] = True
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("_show_about"):
        try:
            # st.dialog is decorator-based and shows a real modal in 1.31+
            @st.dialog("About Predictive Serve", width="large")
            def _show():
                _render_about_dialog()
                if st.button("Close", key="about_close"):
                    st.session_state["_show_about"] = False
                    st.rerun()
            _show()
        except Exception:
            # Older Streamlit: fall back to an inline expander
            with st.expander("About Predictive Serve", expanded=True):
                _render_about_dialog()
                if st.button("Close", key="about_close_inline"):
                    st.session_state["_show_about"] = False
                    st.rerun()


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


def render_coverage() -> None:
    """A collapsible transparency strip listing what's in / out of the model."""
    pred_df = load_predictions()
    n_matches = len(pred_df) if not pred_df.empty else 0
    latest = (
        pd.to_datetime(pred_df["date"]).max().date().strftime("%d %b %Y")
        if not pred_df.empty
        else "—"
    )
    earliest = (
        pd.to_datetime(pred_df["date"]).min().date().strftime("%d %b %Y")
        if not pred_df.empty
        else "—"
    )
    info = load_metrics_json()
    model_label = str(info.get("model", "Model"))
    api_supp_path = PROCESSED_DIR / "recent_results_apitennis.csv"
    api_supp_n = 0
    api_supp_latest = "—"
    if api_supp_path.exists():
        try:
            sup = pd.read_csv(api_supp_path)
            api_supp_n = len(sup)
            if api_supp_n:
                api_supp_latest = pd.to_datetime(sup["date"]).max().strftime("%d %b %Y")
        except Exception:
            pass

    st.markdown(
        f"""
        <details class="ps-details" style="margin: 0 0 14px 0;">
          <summary>Data coverage — what the model trains on</summary>
          <div class="coverage-grid">
            <div class="cov-card cov-card-good">
              <div class="cov-title">✓ Included</div>
              <ul>
                <li>ATP main-tour singles (Grand Slams, Masters 1000, ATP 500, ATP 250, Tour Finals)</li>
                <li>Historical archive from tennis-data.co.uk — <b>{n_matches:,}</b> matches, <b>{earliest}</b> → <b>{latest}</b></li>
                <li>API-Tennis recent results supplement (last 21 days, finished singles) — <b>{api_supp_n}</b> matches up to <b>{api_supp_latest}</b></li>
                <li>Active model: <b>{h(model_label)}</b> (gradient-boosted, 36 leakage-safe features)</li>
              </ul>
            </div>
            <div class="cov-card cov-card-bad">
              <div class="cov-title">✕ Excluded</div>
              <ul>
                <li>Challenger / ITF / Futures circuits</li>
                <li>WTA matches (men's tour only for now)</li>
                <li>Doubles, mixed doubles, wheelchair, junior, exhibition</li>
                <li>Bookmaker odds <i>(used for reference only — kept out of features so the "edge" stays meaningful)</i></li>
              </ul>
            </div>
          </div>
        </details>
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

    sub = sub.sort_values("date")
    y = sub["y"].astype(int).values
    p = sub["p_model"].astype(float).values
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    ac = accuracy_score(y, (p >= 0.5).astype(int))

    # Last 7 days within the selection
    cutoff = sub["date"].max() - pd.Timedelta(days=7)
    last_week = sub[sub["date"] > cutoff]
    if not last_week.empty:
        ywk = last_week["y"].astype(int).values
        pwk = last_week["p_model"].astype(float).values
        wk_acc = accuracy_score(ywk, (pwk >= 0.5).astype(int))
        wk_n = len(last_week)
    else:
        wk_acc, wk_n = None, 0

    # Last 20 picks chronologically
    last20 = sub.tail(20)
    if not last20.empty:
        y20 = last20["y"].astype(int).values
        p20 = last20["p_model"].astype(float).values
        p20_acc = accuracy_score(y20, (p20 >= 0.5).astype(int))
    else:
        p20_acc = None

    headline = [
        _kpi("Predictions", f"{len(sub):,}", f"{sub['date'].min().date()} - {sub['date'].max().date()}"),
        _kpi("Pick accuracy", _fmt_pct(ac), "share of correct AI picks"),
        _kpi("Last 7 days", _fmt_pct(wk_acc), f"on {wk_n:,} matches"),
        _kpi("Last 20 picks", _fmt_pct(p20_acc), f"on {min(20, len(sub)):,} most recent"),
    ]
    st.markdown(f'<div class="ps-kpi-grid">{"".join(headline)}</div>', unsafe_allow_html=True)

    # Secondary metrics (smaller, two-up) for technical readers
    high_mask = (p < 0.35) | (p > 0.65)
    high_n = int(high_mask.sum())
    high_acc = (
        accuracy_score(y[high_mask], (p[high_mask] >= 0.5).astype(int))
        if high_n > 0
        else None
    )
    secondary = [
        _kpi("Log loss", _fmt_num(ll), "lower is better"),
        _kpi("Brier score", _fmt_num(br), "lower is better"),
        _kpi("High-confidence picks", _fmt_pct(high_acc), f"on {high_n:,} confident calls (p<0.35 or p>0.65)"),
    ]
    st.markdown(
        f'<details class="ps-details"><summary>Detailed metrics</summary>'
        f'<div class="ps-kpi-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));">{"".join(secondary)}</div>'
        f'</details>',
        unsafe_allow_html=True,
    )


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
    """Symmetric jump row that mirrors the upcoming card layout:
    Player A button on the left, Tournament centred, Player B on the right."""
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if player_a:
            st.button(
                f"👤 {display_name(player_a)}",
                key=f"{prefix}_pa",
                on_click=navigate_to_player,
                args=(player_a,),
                width="stretch",
                help="Open player profile",
            )
    with cols[1]:
        if tournament:
            st.button(
                f"🏆 {tournament}",
                key=f"{prefix}_t",
                on_click=navigate_to_tournament,
                args=(tournament,),
                width="stretch",
                help="Open tournament profile",
            )
    with cols[2]:
        if player_b:
            st.button(
                f"👤 {display_name(player_b)}",
                key=f"{prefix}_pb",
                on_click=navigate_to_player,
                args=(player_b,),
                width="stretch",
                help="Open player profile",
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

    df = df.sort_values("date", ascending=False).head(500).reset_index(drop=True)
    actual_winner = (
        np.where(df["y"].astype("Int64").fillna(-1).astype(int) == 1, df["playerA"], df["playerB"])
        if "y" in df.columns
        else None
    )
    correct_mask = (df["winner_pick"].values == actual_winner) if actual_winner is not None else None

    # Keep the underlying date as a proper datetime so the Date column sorts
    # chronologically (a text "1 May 2026" sorts before "10 April 2026").
    show = pd.DataFrame({
        "Date": df["date"].dt.date,
        "Surface": df["surface"],
        "Player A": df["playerA"].map(display_name),
        "Player B": df["playerB"].map(display_name),
        "AI Pick": df["winner_pick"].map(display_name),
        "Confidence": df["winner_prob"] * 100.0,
    })
    if actual_winner is not None:
        show["Actual Winner"] = pd.Series(actual_winner, index=df.index).map(display_name)
        show["Result"] = np.where(correct_mask, "✓", "✗")

    column_config = {
        "Date": st.column_config.DateColumn("Date", format="DD MMMM YYYY", width="medium"),
        "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
    }

    if correct_mask is not None:
        styled = _style_correct_rows(show, correct_mask).format({"Confidence": "{:.1f}%"})
        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=520,
            column_config=column_config,
        )
    else:
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            height=520,
            column_config=column_config,
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
            "event_time": (str(r.get("event_time") or "").strip() or None),
            "tournament": r.get("tournament", ""),
            "round": r.get("round", ""),
            "surface": r["surface"],
            "playerA": r["playerA"],
            "playerB": r["playerB"],
            "status": (str(r.get("status") or "").strip().lower() or None),
            "score": (str(r.get("score") or "").strip() or None),
            "winner_side": (str(r.get("winner_side") or "").strip() or None),
            "p_model": p,
            "_source": r.get("_source", "real"),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["winner_prob"] = np.where(out["p_model"] >= 0.5, out["p_model"], 1.0 - out["p_model"])
    out["winner_pick"] = np.where(out["p_model"] >= 0.5, out["playerA"], out["playerB"])
    return out


@st.cache_data(show_spinner=False)
def _history_player_index(history_df: pd.DataFrame) -> Tuple[dict, dict, list]:
    if history_df.empty:
        return {}, {}, []
    names = pd.concat([history_df["playerA"], history_df["playerB"]]).dropna().astype(str).unique().tolist()
    by_init_sur, by_surname = build_history_index(names)
    return by_init_sur, by_surname, names


def tab_upcoming(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Upcoming Fixtures</div>", unsafe_allow_html=True)

    real = load_real_fixtures()
    has_real = not real.empty
    has_key = _api_key() is not None

    if has_real:
        real = real.copy()
        real["date"] = pd.to_datetime(real["date"], errors="coerce")
        real = real.dropna(subset=["date", "playerA", "playerB", "surface"])
        real = real[real["date"] >= pd.Timestamp.today().normalize()]
        if real.empty:
            has_real = False

    # Toolbar
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
            st.caption("Set API_TENNIS_KEY in .env to enable live fixtures.")
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

    fix = real if has_real else synth_fixtures(history_df)
    if fix.empty:
        st.markdown('<div class="empty-state">Could not generate any fixtures (history dataset empty).</div>', unsafe_allow_html=True)
        return

    # Drop doubles (we never trained on doubles), then resolve API-style names
    # ("J. Sinner") to history-style ones ("Sinner J.") so the model finds the
    # right snapshots. Anything we can't resolve is skipped — those are the
    # 50% predictions the user was seeing.
    by_init_sur, by_surname, _hist_names = _history_player_index(history_df)
    fix = fix.copy()
    fix = fix[~fix["playerA"].astype(str).map(is_doubles)]
    fix = fix[~fix["playerB"].astype(str).map(is_doubles)]
    fix["playerA_resolved"] = fix["playerA"].apply(lambda n: resolve_history_name(str(n), by_init_sur, by_surname))
    fix["playerB_resolved"] = fix["playerB"].apply(lambda n: resolve_history_name(str(n), by_init_sur, by_surname))
    n_before = len(fix)
    fix = fix.dropna(subset=["playerA_resolved", "playerB_resolved"]).copy()
    n_after = len(fix)
    fix["playerA"] = fix["playerA_resolved"]
    fix["playerB"] = fix["playerB_resolved"]
    fix = fix.drop(columns=["playerA_resolved", "playerB_resolved"])

    # Default = today + tomorrow only (fast and the only window worth showing
    # most of the time).
    today = pd.Timestamp.today().normalize()
    horizon = today + pd.Timedelta(days=2)

    f1, f2, f3 = st.columns([1.4, 2.4, 1.4])
    with f1:
        days_ahead = st.slider("Days ahead", 1, 7, 2, key="u_days_ahead")
    with f2:
        tour_opts = all_tournaments(fix)
        sel_tours = st.multiselect("Tournament filter (optional)", tour_opts, default=[], key="u_tours")
    with f3:
        min_conf = st.slider("Min confidence", 0.50, 0.95, 0.60, 0.01, key="u_minconf")

    horizon = today + pd.Timedelta(days=days_ahead)
    fix = fix[(fix["date"] >= today) & (fix["date"] < horizon)]
    if sel_tours:
        fix = fix[fix["tournament"].isin(sel_tours)]
    if fix.empty:
        st.markdown(
            '<div class="empty-state">No singles fixtures from our trained roster in this window. '
            'Try expanding the day count or check back later.</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"After resolution: {n_after:,} fixtures from our roster (raw fixtures: {n_before:,}).")
        return

    sc = _score_fixtures(fix, history_df)
    if sc.empty:
        st.markdown(
            '<div class="empty-state">Could not score fixtures (model artifacts or history snapshots missing).</div>',
            unsafe_allow_html=True,
        )
        return

    sc = sc[sc["winner_prob"] >= min_conf]

    # Tag matches currently live (live API players match any pair in our list).
    livescores = cached_livescore()
    live_canon = set()
    for ev in livescores:
        a = ev.get("event_first_player") or ""
        b = ev.get("event_second_player") or ""
        if a and b and not is_doubles(a) and not is_doubles(b):
            live_canon.add(frozenset((canonical_parts(a), canonical_parts(b))))
    sc["is_live"] = sc.apply(
        lambda r: frozenset((canonical_parts(str(r["playerA"])), canonical_parts(str(r["playerB"])))) in live_canon,
        axis=1,
    )

    # Status bucket: live > scheduled > finished
    def _bucket(row):
        if row.get("is_live"):
            return "live"
        status = (row.get("status") or "").lower()
        if "finish" in status or "ended" in status or "walkover" in status:
            return "finished"
        return "scheduled"
    sc["bucket"] = sc.apply(_bucket, axis=1)

    n_live = int((sc["bucket"] == "live").sum())
    n_sched = int((sc["bucket"] == "scheduled").sum())
    n_fin = int((sc["bucket"] == "finished").sum())

    st.markdown(
        f"<div class='ps-section-title'>{len(sc):,} matches &middot; "
        f"<span style='color:var(--muted);font-weight:500;'>"
        f"{n_live} live &middot; {n_sched} scheduled &middot; {n_fin} finished &middot; "
        f"trained roster only ({n_after:,} of {n_before:,} singles)</span></div>",
        unsafe_allow_html=True,
    )

    # ---- LIVE section first ---------------------------------------------
    live_df = sc[sc["bucket"] == "live"].copy()
    if not live_df.empty:
        st.markdown(
            f"<div class='date-group'><div class='day'>"
            f"<span class='dow'>Now</span>Live right now</div>"
            f"<div class='count'>{len(live_df)} match{'es' if len(live_df) != 1 else ''}</div></div>",
            unsafe_allow_html=True,
        )
        for i, (_, r) in enumerate(live_df.iterrows()):
            _render_upcoming_card(r, f"u_live_{i}", state="live")

    # ---- SCHEDULED grouped by day -> tournament, sorted by time ---------
    sched_df = sc[sc["bucket"] == "scheduled"].copy()
    sched_df["__time"] = sched_df["event_time"].fillna("99:99")
    sched_df = sched_df.sort_values(["date", "__time", "tournament"])
    for day, day_group in sched_df.groupby(sched_df["date"].dt.date):
        day_dt = pd.Timestamp(day)
        st.markdown(
            f"<div class='date-group'>"
            f"<div class='day'><span class='dow'>{day_dt.strftime('%a')}</span>"
            f"{fmt_date_long(day_dt)}</div>"
            f"<div class='count'>{len(day_group)} match{'es' if len(day_group) != 1 else ''}</div></div>",
            unsafe_allow_html=True,
        )
        for tour, tour_group in day_group.groupby("tournament", sort=False):
            st.markdown(
                f"<div class='tour-subheader'><span>🏆 {h(str(tour) or 'Unknown')}</span>"
                f"<span class='count'>{len(tour_group)} match{'es' if len(tour_group) != 1 else ''}</span></div>",
                unsafe_allow_html=True,
            )
            for i, (_, r) in enumerate(tour_group.iterrows()):
                _render_upcoming_card(r, f"u_sched_{day}_{slugify(str(tour))}_{i}", state="scheduled")

    # ---- FINISHED at the bottom (newest first) --------------------------
    fin_df = sc[sc["bucket"] == "finished"].copy()
    if not fin_df.empty:
        fin_df = fin_df.sort_values("date", ascending=False)
        st.markdown(
            f"<div class='date-group'><div class='day'>"
            f"<span class='dow'>Done</span>Finished matches</div>"
            f"<div class='count'>{len(fin_df)} match{'es' if len(fin_df) != 1 else ''}</div></div>",
            unsafe_allow_html=True,
        )
        for i, (_, r) in enumerate(fin_df.iterrows()):
            _render_upcoming_card(r, f"u_fin_{i}", state="finished")


# =============================================================================
# Tab: Players
# =============================================================================

def _player_history(history_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Pull every match for *player*, including spellings that share its
    canonical (initial, surname) — so career stats aren't split across
    'Carreno Busta P.' / 'Carreno-Busta P.'."""
    if history_df.empty:
        return pd.DataFrame()
    target = _canonical_key(player)
    variants = _name_variants_index(history_df).get(target, [player])
    variants_set = set(variants)
    mask = history_df["playerA"].isin(variants_set) | history_df["playerB"].isin(variants_set)
    h_ = history_df[mask].copy().sort_values("date")
    if h_.empty:
        return h_
    h_["is_winner"] = h_["playerA"].isin(variants_set).astype(int)
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

    search = st.text_input(
        "Search a player",
        value="",
        placeholder="Type a name (full or partial) — e.g. Sinner, Carlos, Roger",
        key="p_search",
        label_visibility="visible",
    ).strip().lower()

    flt = directory[directory["matches"] >= 30].copy()
    if search:
        # Match either the history short form OR the cached full name.
        cache = _load_player_cache()
        full_names = {n: (cache[n].full_name or "") for n in flt["player"] if n in cache}
        def _haystack(player_key: str) -> str:
            return f"{player_key} {full_names.get(player_key, '')}".lower()
        mask = flt["player"].astype(str).apply(lambda n: search in _haystack(n))
        flt = flt[mask]

    # Sort: active players (last_year >= year_max - 1) first, then by WR
    # (eligible only above MIN_WR_MATCHES so a one-game wonder doesn't beat
    # career WRs). Inactive legends fall through after.
    if not flt.empty:
        active_cutoff = int(flt["last_year"].max()) - 1
        flt["is_active"] = (flt["last_year"] >= active_cutoff).astype(int)
        flt["sort_winrate"] = flt["winrate"].where(flt["matches"] >= 50, 0.0)
        flt = flt.sort_values(
            ["is_active", "sort_winrate", "matches"],
            ascending=[False, False, False],
        ).drop(columns=["is_active", "sort_winrate"])

    if flt.empty:
        st.markdown('<div class="empty-state">No players match the current filters.</div>', unsafe_allow_html=True)
        return

    options = flt.head(400).copy()
    label_for = {
        row["player"]: f"{player_label(row['player'])}  -  {int(row['matches']):,} matches  ·  {(row['winrate']*100):.1f}% WR  ·  {int(row['first_year'])}-{int(row['last_year'])}"
        for _, row in options.iterrows()
    }
    keys = list(options["player"])
    pre = st.session_state.get("profile_player")
    default_idx = keys.index(pre) if pre in keys else 0

    player = st.selectbox(
        f"Select a player ({len(flt):,} match the filters, sorted active first by WR)",
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

    # Best-effort metadata from API-Tennis (cached). Renders flag + age in header.
    meta = get_player_meta(player)
    flag_html = (
        f'<span class="flag">{meta.flag}</span>'
        if meta and meta.flag and meta.flag != "🏳️"
        else ""
    )
    profile_name = (meta.full_name or player) if meta else player

    chips: list[str] = []
    if meta and meta.country:
        chips.append(f'<span class="chip">{h(meta.country)}</span>')
    if meta and meta.age:
        chips.append(f'<span class="chip">{meta.age} years</span>')
    chips.append(f'<span class="chip">{first_season}-{last_season}</span>')
    chips.append(f'<span class="chip">{total:,} matches</span>')
    chips.append(f'<span class="chip">{winrate*100:.1f}% career WR</span>')

    img_html = player_image_html(player, size=120)
    st.markdown(
        f"""
        <div class="profile-header">
          <div class="avatar">{img_html}</div>
          <div class="meta-block">
            <div class="name">{flag_html}{h(profile_name)}</div>
            <div class="sub">{wins:,} wins · {losses:,} losses</div>
            <div class="meta-chips">{''.join(chips)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # No manual buttons — metadata + photos auto-resolve on profile open
    # (and are batch-refreshed nightly by the GitHub Actions cron). See
    # src/data/fetch_player_roster.py + fetch_player_photos.py.

    _render_upcoming_for_player(player, history_df)

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
        "Date": recent["date"].map(fmt_date_long),
        "Tournament": recent.get("tournament", ""),
        "Surface": recent["surface"],
        "Round": recent.get("round", ""),
        "Result": np.where(recent["is_winner"] == 1, "Win", "Loss"),
        "Opponent": recent["opponent"].map(display_name),
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

    search = st.text_input(
        "Search a tournament",
        value="",
        placeholder="Type a tournament — e.g. Wimbledon, Roland Garros, Cincinnati",
        key="t_search",
        label_visibility="visible",
    ).strip().lower()

    flt = counts[counts["matches"] >= 30].copy()
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

    surface_badges = "".join(f'<span class="badge">{h(s)}</span>' for s in surfaces)

    st.markdown(
        f"""
        <div class="tour-hero">
          <div>
            <div class="name">🏆 {h(tour)}</div>
            <div class="sub">{first_season} - {latest_season} &middot; {len(years)} editions &middot; {len(sub):,} matches recorded</div>
            <div class="badges">{surface_badges}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Most recent edition: keep only that season's matches, then cluster
    # them by round so the bracket flow reads top-down (R128 -> F).
    latest_year = int(max(years)) if years else None
    if latest_year:
        focus = sub[sub["date"].dt.year == latest_year].copy()
        focus_label = f"{latest_year} edition"
    else:
        focus = sub.copy()
        focus_label = "Most recent matches"

    st.markdown(
        f"<div class='ps-section-title'>{h(focus_label)} &middot; "
        f"<span style='color:var(--muted);font-weight:500;'>by round</span></div>",
        unsafe_allow_html=True,
    )

    if focus.empty:
        st.markdown('<div class="empty-state">No matches in the latest edition.</div>', unsafe_allow_html=True)
        return

    round_col = focus.get("round", pd.Series("", index=focus.index)).astype(str).str.strip().str.upper()
    focus["round_norm"] = round_col.where(round_col != "", "—")
    ROUND_ORDER = ["R128", "R64", "R56", "R48", "R32", "RR", "R16", "QF", "SF", "F"]
    def _r_rank(r):
        try:
            return ROUND_ORDER.index(r)
        except ValueError:
            return -1  # unknowns first

    for round_name in sorted(focus["round_norm"].unique(), key=_r_rank, reverse=True):
        chunk = focus[focus["round_norm"] == round_name].sort_values("date")
        nice = {
            "F": "Final", "SF": "Semifinals", "QF": "Quarterfinals",
            "R16": "Round of 16", "R32": "Round of 32", "R64": "Round of 64",
            "R128": "Round of 128", "RR": "Round Robin",
        }.get(round_name, round_name)
        st.markdown(
            f"<div class='tour-subheader'><span>🎾 {h(nice)}</span>"
            f"<span class='count'>{len(chunk)} match{'es' if len(chunk) != 1 else ''}</span></div>",
            unsafe_allow_html=True,
        )
        cache = _load_player_cache()
        for i, (_, m) in enumerate(chunk.iterrows()):
            pa = str(m["playerA"])
            pb = str(m["playerB"])
            winner = display_name(pa)
            loser = display_name(pb)
            score = str(m.get("score") or "")
            date_s = fmt_date_long(m["date"])
            img_w = player_image_html(pa, size=72)
            img_l = player_image_html(pb, size=72)
            meta_w = cache.get(resolve_to_history_name(pa))
            meta_l = cache.get(resolve_to_history_name(pb))
            flag_w = (meta_w.flag if meta_w and meta_w.flag and meta_w.flag != "🏳️" else "")
            flag_l = (meta_l.flag if meta_l and meta_l.flag and meta_l.flag != "🏳️" else "")
            country_w = (meta_w.country if meta_w and meta_w.country else "")
            country_l = (meta_l.country if meta_l and meta_l.country else "")
            sub_w = f"{flag_w}&nbsp;{h(country_w)}" if country_w else ""
            sub_l = f"{flag_l}&nbsp;{h(country_l)}" if country_l else ""
            st.markdown(
                f"""
                <div class="up-card up-card-finished" style="border-left:3px solid var(--good);">
                  <div class="up-meta">
                    <span class="fin-pill-card">FINAL</span>
                    <span class="up-time">{h(date_s)}</span>
                  </div>
                  <div class="up-row">
                    <div class="up-side">
                      <div class="up-photo round-winner-photo">{img_w}</div>
                      <div class="up-info">
                        <div class="up-name">{h(winner)} <span class="round-badge-w">WIN</span></div>
                        <div class="up-prob" style="color:var(--muted);">{sub_w or '&nbsp;'}</div>
                      </div>
                    </div>
                    <div class="up-pick">
                      <div class="up-pick-label">FINAL SCORE</div>
                      <div class="up-pick-name" style="font-size:1.35rem;">{h(score) or '—'}</div>
                    </div>
                    <div class="up-side up-side-right">
                      <div class="up-info up-info-right">
                        <div class="up-name">{h(loser)}</div>
                        <div class="up-prob" style="color:var(--muted);">{sub_l or '&nbsp;'}</div>
                      </div>
                      <div class="up-photo">{img_l}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
        row["player"]: f"{player_label(row['player'])}  -  {int(row['matches']):,} matches  ·  {(row['winrate']*100):.1f}% WR"
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

    render_h2h(pa, pb)

    cols = st.columns([1, 1, 4])
    with cols[0]:
        st.button(f"👤 {display_name(pa)}", key="w_open_pa", on_click=navigate_to_player, args=(pa,), width="stretch")
    with cols[1]:
        st.button(f"👤 {display_name(pb)}", key="w_open_pb", on_click=navigate_to_player, args=(pb,), width="stretch")


# =============================================================================
# Tab: Leaderboard
# =============================================================================

def _leaderboard_table(df: pd.DataFrame, pred_df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """
    Build a richer leaderboard with:
      - Matches / Wins / Losses / Win rate
      - Last 30 days win rate
      - Best surface
      - AI accuracy on this player's matches (from predictions)
      - Current win streak (positive = wins, negative = losses)
    """
    if df.empty:
        return pd.DataFrame()

    # Wins / losses
    wins = df["playerA"].value_counts()
    losses = df["playerB"].value_counts()
    base = pd.DataFrame({"Wins": wins, "Losses": losses}).fillna(0).astype(int)
    base["Matches"] = base["Wins"] + base["Losses"]
    base = base[base["Matches"] >= int(min_matches)]
    if base.empty:
        return base
    base["Win rate"] = (base["Wins"] / base["Matches"].replace(0, np.nan)) * 100

    # Last 30 days win rate
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df[df["date"] >= cutoff]
    rwins = recent["playerA"].value_counts()
    rlosses = recent["playerB"].value_counts()
    rd = pd.DataFrame({"rw": rwins, "rl": rlosses}).fillna(0)
    rd["L30 WR"] = (rd["rw"] / (rd["rw"] + rd["rl"]).replace(0, np.nan)) * 100
    base["L30 WR"] = rd["L30 WR"]

    # Best surface
    surface_winrate: dict[str, str] = {}
    grouped = df.groupby(["playerA", "surface"]).size().rename("w").reset_index()
    grouped_l = df.groupby(["playerB", "surface"]).size().rename("l").reset_index()
    surf_w = grouped.pivot(index="playerA", columns="surface", values="w").fillna(0)
    surf_l = grouped_l.pivot(index="playerB", columns="surface", values="l").fillna(0)
    surf_total = surf_w.add(surf_l, fill_value=0)
    surf_wr = surf_w.div(surf_total.replace(0, np.nan)) * 100
    for player in base.index:
        if player in surf_wr.index:
            row = surf_wr.loc[player].dropna()
            row = row[surf_total.loc[player] >= 10] if player in surf_total.index else row
            if not row.empty:
                top_surface = row.idxmax()
                surface_winrate[player] = f"{top_surface} ({row.max():.0f}%)"
    base["Best surface"] = base.index.map(surface_winrate.get)

    # AI accuracy on this player's matches (from predictions)
    if not pred_df.empty and "p_model" in pred_df.columns and "y" in pred_df.columns:
        p = pred_df.dropna(subset=["p_model", "y"]).copy()
        p["pick"] = np.where(p["p_model"] >= 0.5, p["playerA"], p["playerB"])
        p["actual"] = np.where(p["y"].astype(int) == 1, p["playerA"], p["playerB"])
        p["correct"] = (p["pick"] == p["actual"]).astype(int)
        # Player appears as A or B
        for_a = p[p["playerA"].isin(base.index)].groupby("playerA")["correct"].agg(["sum", "count"])
        for_b = p[p["playerB"].isin(base.index)].groupby("playerB")["correct"].agg(["sum", "count"])
        combined = for_a.add(for_b, fill_value=0)
        combined["AI Acc"] = (combined["sum"] / combined["count"].replace(0, np.nan)) * 100
        base["AI accuracy"] = combined["AI Acc"]

    # Current streak
    streaks: dict[str, int] = {}
    df_sorted = df.sort_values("date")
    for player in base.index:
        runs = df_sorted[(df_sorted["playerA"] == player) | (df_sorted["playerB"] == player)]
        if runs.empty:
            continue
        # Walk from latest match backwards, counting consecutive same-result matches
        outcomes = (runs["playerA"] == player).astype(int).values  # 1 = win, 0 = loss
        last = outcomes[-1]
        streak = 0
        for o in outcomes[::-1]:
            if o == last:
                streak += 1
            else:
                break
        streaks[player] = streak if last == 1 else -streak
    base["Streak"] = base.index.map(streaks.get)

    base.index.name = "Player"
    return base.reset_index()


def tab_leaderboard(history_df: pd.DataFrame) -> None:
    st.markdown("<div class='ps-section-title'>Leaderboard</div>", unsafe_allow_html=True)
    if history_df.empty:
        st.markdown('<div class="empty-state">No history dataset available.</div>', unsafe_allow_html=True)
        return

    surf_opts = sorted(history_df["surface"].dropna().unique().tolist())
    year_min = int(history_df["date"].dt.year.dropna().min())
    year_max = int(history_df["date"].dt.year.dropna().max())

    f1, f2, f3, f4, f5 = st.columns([1.2, 1.8, 1.8, 1.8, 1])
    with f1:
        min_matches = st.number_input("Min matches", min_value=10, max_value=1000, value=80, step=10, key="l_min")
    with f2:
        if year_min < year_max:
            year_range = st.slider(
                "Window",
                min_value=year_min,
                max_value=year_max,
                value=(max(year_min, year_max - 2), year_max),
                key="l_years",
            )
        else:
            year_range = (year_min, year_max)
    with f3:
        sel_surfaces = st.multiselect("Surface", surf_opts, default=surf_opts, key="l_surfaces")
    with f4:
        sort_by = st.selectbox(
            "Rank by",
            ["Win rate", "Wins", "L30 WR", "AI accuracy", "Streak"],
            index=0,
            key="l_sort",
        )
    with f5:
        top_n = st.slider("Top", 10, 200, 50, key="l_topn")

    df = history_df.copy()
    df = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])]
    if sel_surfaces:
        df = df[df["surface"].isin(sel_surfaces)]
    if df.empty:
        st.markdown('<div class="empty-state">No matches in the current selection.</div>', unsafe_allow_html=True)
        return

    pred_df = load_predictions()
    pred_window = pred_df.copy()
    if not pred_window.empty:
        pred_window = pred_window[(pred_window["date"].dt.year >= year_range[0]) & (pred_window["date"].dt.year <= year_range[1])]
        if sel_surfaces:
            pred_window = pred_window[pred_window["surface"].isin(sel_surfaces)]

    lb = _leaderboard_table(df, pred_window, min_matches)
    if lb.empty:
        st.markdown('<div class="empty-state">No players meet the minimum match threshold.</div>', unsafe_allow_html=True)
        return

    # Sort and rank
    sort_col_map = {
        "Win rate": "Win rate",
        "Wins": "Wins",
        "L30 WR": "L30 WR",
        "AI accuracy": "AI accuracy",
        "Streak": "Streak",
    }
    sc = sort_col_map[sort_by]
    lb = lb.sort_values([sc, "Wins"], ascending=[False, False], na_position="last").head(int(top_n))
    lb.insert(0, "Rank", np.arange(1, len(lb) + 1))

    def _fmt_streak(s):
        if pd.isna(s):
            return ""
        s = int(s)
        if s == 0:
            return ""
        return f"{'W' if s > 0 else 'L'}{abs(s)}"

    lb["StreakLabel"] = lb["Streak"].apply(_fmt_streak)
    for c in ("Wins", "Losses", "Matches"):
        lb[c] = lb[c].astype(int)

    # Render rich cards. iterrows() preserves the original column names
    # (itertuples mangles "Win rate" into "_5" / "Win_rate" depending on
    # pandas version, which is what hid every stat except Streak earlier).
    cache = _load_player_cache()
    for i, (_, row) in enumerate(lb.iterrows()):
        rank = int(row["Rank"])
        player = str(row["Player"])
        full = display_name(player)
        meta = cache.get(player)
        flag = (meta.flag if meta and meta.flag and meta.flag != "🏳️" else "")
        country = (meta.country if meta and meta.country else "")
        photo = player_image_html(player, size=64)

        def _val(key):
            v = row.get(key)
            return v if pd.notna(v) else None

        wr = _val("Win rate")
        l30 = _val("L30 WR")
        ai = _val("AI accuracy")
        best_surface = _val("Best surface") or ""
        streak_lbl = row.get("StreakLabel") or ""
        streak_class = "good" if streak_lbl.startswith("W") else "bad" if streak_lbl.startswith("L") else ""

        wr_html = f"{wr:.1f}%" if wr is not None else "-"
        l30_html = f"{l30:.1f}%" if l30 is not None else "-"
        ai_html = f"{ai:.1f}%" if ai is not None else "-"

        st.markdown(
            f"""
            <div class="lb-card">
              <div class="lb-rank">#{rank}</div>
              <div class="lb-photo">{photo}</div>
              <div class="lb-meta">
                <div class="lb-name">{flag} {h(full)}</div>
                <div class="lb-sub">{h(country) or '&nbsp;'} &middot; {int(row['Matches']):,} matches &middot; {int(row['Wins']):,}-{int(row['Losses']):,} W-L</div>
              </div>
              <div class="lb-stat"><div class="lb-stat-label">Win rate</div><div class="lb-stat-val">{wr_html}</div></div>
              <div class="lb-stat"><div class="lb-stat-label">Last 30d</div><div class="lb-stat-val">{l30_html}</div></div>
              <div class="lb-stat"><div class="lb-stat-label">AI Acc</div><div class="lb-stat-val">{ai_html}</div></div>
              <div class="lb-stat"><div class="lb-stat-label">Surface</div><div class="lb-stat-val small">{h(best_surface) or '-'}</div></div>
              <div class="lb-stat"><div class="lb-stat-label">Streak</div><div class="lb-stat-val streak-{streak_class}">{streak_lbl or '-'}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            f"👤 View profile",
            key=f"lb_open_{i}",
            on_click=navigate_to_player,
            args=(player,),
            width="stretch",
        )


# =============================================================================
# Main
# =============================================================================

_init_state()
render_nav()
render_live_ticker()
render_hero()
render_coverage()

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
