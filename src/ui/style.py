"""Stylesheet for the Predictive Serve Streamlit app.

Lives in a dedicated module so the entry-point streamlit_app.py stays
focused on app logic. Imported once and injected via st.markdown.
"""
from __future__ import annotations

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
  margin: 6px 0 20px 0;
  padding: 32px 34px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--line);
  background:
    radial-gradient(900px 280px at 8% -25%, rgba(78,161,255,0.22), transparent 55%),
    radial-gradient(700px 260px at 95% -10%, rgba(255,122,61,0.16), transparent 60%),
    radial-gradient(500px 200px at 50% 110%, rgba(124,92,255,0.10), transparent 70%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.018));
  position: relative; overflow: hidden;
}
.ps-hero::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 40%);
  border-radius: inherit;
}
.ps-eyebrow {
  display:inline-flex; gap:6px; padding:6px 13px; border-radius:999px;
  background: rgba(78,161,255,0.12); border:1px solid rgba(78,161,255,0.32);
  color:#b8d6ff; font-size:0.73rem; font-weight:700;
  letter-spacing:0.08em; text-transform:uppercase;
}
.ps-hero h1 {
  margin: 18px 0 10px 0;
  font-size: clamp(1.85rem, 3.2vw, 2.6rem);
  line-height:1.05; font-weight:800; letter-spacing:-0.025em; color:#fff;
}
.ps-hero h1 span.grad {
  background: linear-gradient(95deg, #ffffff 0%, #c4def0 45%, #ffd0bf 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.ps-hero p {
  margin:0; color:var(--muted); font-size:1.02rem; max-width:820px;
  line-height:1.55;
}

/* KPI grid */
.ps-kpi-grid {
  display:grid; grid-template-columns: repeat(4, minmax(0,1fr));
  gap:12px; margin: 10px 0 18px 0;
}
@media (max-width: 1100px) { .ps-kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.ps-kpi {
  padding:16px 18px; border-radius: var(--radius);
  border:1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  position:relative; overflow:hidden;
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.22s ease;
}
.ps-kpi:hover {
  transform: translateY(-2px);
  border-color: var(--line-strong);
  box-shadow: 0 14px 30px -16px rgba(106,169,255,0.35);
}
.ps-kpi::after {
  content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); opacity:.6;
}
.ps-kpi-label { font-size:0.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.ps-kpi-val {
  font-size:1.65rem; font-weight:800; color:#fff; margin-top:6px;
  letter-spacing:-0.02em; line-height:1.1;
  font-variant-numeric: tabular-nums;
}
.ps-kpi-sub { font-size:0.78rem; color:var(--muted); margin-top:4px; }

/* Cards */
.ps-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  border:1px solid var(--line); border-radius: var(--radius);
  padding:18px; margin-bottom:14px;
}
.ps-card.tight { padding:12px 14px; }
.ps-section-title {
  font-size:1.08rem; font-weight:700; color:#fff;
  letter-spacing:-0.015em; margin:14px 0 12px 0;
  display: flex; align-items: baseline; gap: 10px;
}
.ps-section-title::before {
  content: "";
  display: inline-block;
  width: 4px; height: 14px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
  border-radius: 2px;
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
  transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.22s ease;
}
.up-card:hover {
  border-color: rgba(106,169,255,0.32);
  transform: translateY(-2px);
  box-shadow: 0 18px 36px -22px rgba(106,169,255,0.45);
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

/* Tale of the tape */
.tot-head, .tot-row {
  display: grid; grid-template-columns: 1fr 1.4fr 1fr;
  align-items: center; gap: 12px;
}
.tot-head { padding: 4px 0 8px 0; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.tot-row { padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.tot-row:last-child { border-bottom: none; }
.tot-val {
  text-align: center; font-weight: 700; font-size: 1.0rem;
  color: var(--muted); font-variant-numeric: tabular-nums;
}
.tot-val.tot-a, .tot-val.tot-b { color: #fff; }
.tot-val.tot-a::after { content: " ◂"; color: var(--accent); font-size: 0.8rem; }
.tot-val.tot-b::before { content: "▸ "; color: var(--accent); font-size: 0.8rem; }
.tot-label {
  text-align: center; color: var(--soft-muted);
  font-size: 0.78rem; letter-spacing: 0.03em; text-transform: uppercase; font-weight: 600;
}

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

/* Kelly bet sizing pills */
.kelly-row { margin: 4px 0 10px 0; display: flex; gap: 8px; flex-wrap: wrap; }
.kelly-pill {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 700;
  border: 1px solid;
  font-variant-numeric: tabular-nums;
}
.kelly-pill.kelly-a { background: rgba(255, 112, 89, 0.12); color: #ffb27a; border-color: rgba(255, 112, 89, 0.30); }
.kelly-pill.kelly-b { background: rgba(106, 169, 255, 0.12); color: #9fc4ff; border-color: rgba(106, 169, 255, 0.30); }

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
.round-winner-photo > div, .round-winner-photo > img {
  border-color: rgba(45, 210, 154, 0.55) !important;
  box-shadow: 0 0 14px rgba(45, 210, 154, 0.25);
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
