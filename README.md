# Predictive Serve 🎾

**Predictive Serve** is an end-to-end Python application for **professional tennis match
forecasting**. It combines a reproducible historical data pipeline, leakage-safe feature
engineering, a calibrated gradient-boosted model, and a polished Streamlit console that
explores predictions, live upcoming fixtures, and per-player / per-tournament profiles.

---

## What it does

- **Historical pipeline** — downloads ATP match data from [tennis-data.co.uk](http://www.tennis-data.co.uk)
  (seasons 2000-present) and rebuilds a clean, deduplicated match table.
- **Feature engineering** — global + surface Elo, short- and long-term form, head-to-head,
  workload, surface, round importance, and tournament tier.
- **Production model** — `HistGradientBoostingClassifier` selected against a Logistic
  Regression baseline, with optional Platt / Isotonic calibration on a within-train slice.
  Market-derived features are **excluded by design** so the AI signal is independent of
  bookmaker prices.
- **Time-aware splits** — train (year < 2022), validation (2022-2024), held-out test
  (year ≥ 2025). Metrics are persisted to `models/metrics.json`.
- **Live upcoming fixtures** — fetched from [api-tennis.com](https://api-tennis.com) with
  intelligent surface inference and median consensus odds across books.
- **Streamlit console** — six views (Matches, Upcoming, Players, Tournaments, What-if,
  Leaderboard) with click-through navigation between them.

---

## Screenshots & feature highlights

- 🟢 **Matches Explorer** — historical predictions with pastel green / red row tinting
  showing which calls the AI got right.
- 📅 **Upcoming** — live fixtures grouped by day, with model probabilities and confidence
  pills. Falls back to a labelled demo dataset when no API key is configured.
- 👤 **Players** — searchable / filterable roster sorted by career match count, full Elo
  trajectory chart, surface breakdown, and on-demand player photo download via API-Tennis.
- 🏆 **Tournaments** — per-event volume, recent champions list, and all-time matches.
- 🎲 **What-if** — pick any two players, surface, round, and date; the model produces a
  win-probability split with two-way pacing bars.
- 🏅 **Leaderboard** — windowed win-rate ranking with per-surface filters.

Every view contains "Open profile" jumps so a player or tournament name in any context can
take you to its full profile in one click.

---

## Quick start

### Prerequisites

- **Python 3.10+** (the project is tested on 3.11)
- On Windows, the [`py` launcher](https://docs.python.org/3/using/windows.html#getting-started)
  is recommended.

### One-click launch (Windows)

Double-click [`run_predictive_serve.bat`](./run_predictive_serve.bat). The script will:

1. Install dependencies from [`requirements.txt`](./requirements.txt)
2. Download historical match data
3. Run the data + feature pipeline
4. Train and evaluate the model
5. Score every match in the dataset
6. (Optional) fetch live upcoming fixtures + odds via API-Tennis
7. Open the Streamlit UI on `http://localhost:8501`

### Manual setup (any OS)

```bash
pip install -r requirements.txt

# 1) Historical pipeline
py -m src.data.fetch_data
py -m src.data.preprocess
py -m src.data.cleaning

# 2) Feature engineering
py -m src.features.elo
py -m src.features.form
py -m src.features.sets
py -m src.features.build_features

# 3) Model selection + held-out test evaluation
py -m src.models.train_best

# 4) Score every match for the UI
py -m src.models.score_all_matches

# 5) (Optional) live fixtures + odds
py -m src.data.fetch_upcoming_apitennis
py -m src.data.fetch_odds_apitennis

# 6) UI
py -m streamlit run streamlit_app.py
```

---

## API-Tennis configuration

The Upcoming and Players (live photo) views are powered by
[api-tennis.com](https://api-tennis.com). To enable them:

```bash
cp .env.example .env
# then edit .env and set:
#   API_TENNIS_KEY=<your-key>
```

The repository does not commit `.env` (it's listed in `.gitignore`). All API responses are
cached on disk under `data/cache/api_tennis/` to respect the provider's rate limits — the
TTL is configurable via `API_TENNIS_CACHE_TTL_S` (fixtures) and
`API_TENNIS_ODDS_CACHE_TTL_S` (odds) in `.env`.

When a key is configured, the Streamlit UI shows a "Refresh fixtures from API" button in
the Upcoming view. Otherwise the view degrades gracefully to a labelled demo dataset
synthesized from recent active players.

---

## Architecture

```
┌──────────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│ tennis-data.co.uk        │    │ api-tennis.com         │    │ models/             │
│ historical Excel files   │    │ live fixtures + odds   │    │ logreg_final.pkl    │
└────────────┬─────────────┘    └────────────┬───────────┘    │ imputer_final.pkl   │
             │                               │                │ feature_columns.txt │
             ▼                               ▼                │ metrics.json        │
┌──────────────────────────┐    ┌────────────────────────┐    └─────────────────────┘
│ src/data/                │    │ src/integrations/      │              ▲
│   fetch_data → CSV       │    │   api_tennis (cache)   │              │
│   preprocess → schema    │    │   surface inference    │              │
│   cleaning → matches     │    │   consensus odds       │              │
└────────────┬─────────────┘    └────────────┬───────────┘              │
             ▼                               ▼                          │
┌──────────────────────────┐    ┌────────────────────────┐              │
│ src/features/            │    │ data/processed/        │              │
│   elo, form, sets        │──▶│  fixtures_upcoming.csv │              │
│   build_features → X     │    │  matches_*.csv          │              │
└────────────┬─────────────┘    │  train_dataset.csv     │              │
             ▼                  │  all_predictions.csv   │              │
┌──────────────────────────┐    └────────────┬───────────┘              │
│ src/models/              │                 │                          │
│   train_best → model     │─────────────────┴──────────────────────────┘
│   score_all_matches      │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ streamlit_app.py         │  ← English-only console
│  6 views, click-through  │
│  player / tournament nav │
└──────────────────────────┘
```

---

## Modelling: leakage-safe by design

The most common pitfall in tennis forecasting is letting bookmaker prices leak into the
model. Predictive Serve takes a deliberate stance:

- The list of **leaky market columns** (`oddsA`, `oddsB`, `pA_market`, `pB_market`,
  `p_diff`, `logit_pA_market`, `has_market`) is centralised in
  [`src/utils/feature_utils.py`](./src/utils/feature_utils.py) and excluded from the
  feature set in **every** training and scoring path.
- The "edge" displayed in the UI (model − market) therefore reflects an **independent**
  AI signal rather than a function of the market it is being compared to.
- A held-out test split (year ≥ 2025) is **never** consulted during model selection or
  calibration.

The training script ([`src/models/train_best.py`](./src/models/train_best.py)) writes
[`models/metrics.json`](./models/metrics.json) with both the validation and the held-out
test scores, so changes can be tracked across commits.

---

## UI design

- Dark slate / navy theme with high-contrast typography.
- Sticky top navigation pill that surfaces the live model name + accuracy.
- Six-tile KPI bar (predictions, pick accuracy, log loss, high-confidence accuracy).
- Button-based section navigation that supports **deep linking** — every player and
  tournament name in tables or cards can jump to a profile page.
- Match rows in the explorer are tinted **pastel green** when the AI was correct and
  **pastel red** when it was wrong, with bold "AI Pick" / "Result" columns for fast
  scanning.

---

## Project layout

```
predictive-serve/
├─ data/
│  ├─ raw/             # tennis-data.co.uk merged Excel files
│  └─ processed/       # cleaned matches + feature matrix + predictions
├─ models/             # trained model, imputer, feature list, metrics.json
├─ assets/players/     # cached player photos (downloaded on demand)
├─ src/
│  ├─ data/            # fetchers, preprocess, cleaning, schema
│  ├─ features/        # Elo, form, sets, head-to-head, market features
│  ├─ models/          # train_best, train_logreg, score_all_matches
│  ├─ integrations/    # API-Tennis client (cached + consensus odds)
│  ├─ predict/         # whatif single-match scoring
│  └─ utils/           # paths, env, aliases, feature_utils, surface, avatars
├─ streamlit_app.py    # Streamlit console
├─ run_predictive_serve.bat   # Windows one-click launcher
└─ requirements.txt
```

---

## Development

### Useful commands

```bash
# Re-train and write metrics.json
py -m src.models.train_best

# Sanity check
py test_quick.py
py test_system.py
```

### Roadmap

- Player nationality / age via API-Tennis player endpoint
- Reliability diagram and per-surface calibration plots
- SHAP-style "why this pick" explanations on every match card
- Filter persistence in URL query parameters
- Optional light-mode theme

Contributions welcome — open an issue to discuss bigger changes first.

---

## License

This project is intended for learning, portfolio, and research use. Tennis match results
remain the property of their original sources (tennis-data.co.uk, api-tennis.com).
