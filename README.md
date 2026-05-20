# Predictive Serve 🎾

**Predictive Serve** is a live, self-updating tennis forecasting console.
An end-to-end Python pipeline ingests two decades of ATP data, engineers
~70 leakage-safe features, trains a calibrated gradient-boosted model
against a held-out 2025+ test split, blends the result with the bookmaker
prior, and serves it all through a Streamlit UI with a live-score ticker,
upcoming-fixture cards and full per-player profiles.

A GitHub Actions cron refreshes everything every night at 04:00 UTC, so
the deployed app is always within ~24 hours of the latest match.

---

## What it does

- **Historical pipeline** — downloads ATP match data from
  [tennis-data.co.uk](http://www.tennis-data.co.uk) (seasons 2000-present)
  and rebuilds a clean, deduplicated match table.
- **Live recent-results supplement** — pulls finished singles matches
  from the last 21 days off [api-tennis.com](https://api-tennis.com) and
  merges them into the historical archive, with a tour-level filter that
  drops qualifying / ITF / junior matches (both players must have ≥ 20
  prior main-tour matches).
- **Feature engineering (76 columns → 69 leakage-safe model inputs)**:
  - **Elo**: global + per-surface + Grand-Slam-only tier-specific Elo,
    with **538-style margin-of-victory weighting** and a **dynamic
    K-factor** that decays from ~50 for debutants to ~27 for veterans.
  - **Elo momentum** + signed **win streak** + **recency-weighted last-10
    form** (linear decay).
  - **Head-to-head**: overall + per-surface H2H win-rate counters.
  - **Common opponent**: transitive "if A beat C and B lost to C" signal
    aggregated over every shared opponent.
  - **Set + tiebreak features**: career set win-rate, deciding-set
    win-rate, career tiebreak win-rate — parsed straight from the score
    string.
  - **Context**: round importance, tournament tier (Grand Slam → 4.0,
    Masters 1000 → 3.0, ATP 500 → 2.0, ATP 250 → 1.5), best-of-5 flag,
    surface one-hots, rank, rest days, workload.
- **Model selection** — `HistGradientBoosting`, `LightGBM`, `LogReg` and
  a soft-voting ensemble of all three are evaluated against a 2022-2024
  validation split. Best wins on validation log-loss. Calibration is
  attempted but only kept when it improves log-loss.
- **Per-year market-prior blend** — α is fit separately for each
  validation season, so the blend tracks how good the bookmaker is in
  recent years vs older ones. Live predictions and test scoring use the
  most recent season's α.
- **No data leakage by construction** —
  [`src/utils/feature_utils.py`](./src/utils/feature_utils.py) owns the
  single `LEAKY_MARKET_COLS` allow-list. `select_model_features` enforces
  it in every training and scoring path. The "edge" displayed in the UI
  is therefore a real, independent model-vs-market signal.

### Current metrics (latest pipeline)

| Metric | Pure AI | **Market-aware blend** | Bookmaker val baseline |
|---|---|---|---|
| Validation log loss | 0.6033 | **0.5831** | 0.5829 |
| Validation accuracy | 66.06 % | **68.21 %** | 68.12 % |
| Held-out 2025 test log loss | 0.6089 | **0.5899** | — |
| **Held-out 2025 test accuracy (3,602 matches)** | 66.10 % | **🏆 68.13 %** | — |

The blend is what the UI shows by default — it beats the pure bookmaker
baseline on the held-out test split.

---

## Quick start

### Prerequisites

- **Python 3.11+** (tested on 3.11.x)
- On Windows, the [`py` launcher](https://docs.python.org/3/using/windows.html)
  is recommended.

### One-click launch (Windows)

Double-click [`run_predictive_serve.bat`](./run_predictive_serve.bat).
The script installs dependencies, runs the full data + feature + train +
score pipeline, fetches live fixtures + player photos via API-Tennis,
then opens the Streamlit UI on `http://localhost:8501`.

### Manual setup (any OS)

```bash
pip install -r requirements.txt

# 1) Historical pipeline
py -m src.data.fetch_data
py -m src.data.preprocess
py -m src.data.cleaning

# 2) (Optional) Supplement with finished API-Tennis matches
py -m src.data.fetch_recent_results_apitennis
py -m src.data.merge_recent_results

# 3) Feature engineering + model
py -m src.features.elo
py -m src.features.form
py -m src.features.build_features
py -m src.models.train_best
py -m src.models.score_all_matches

# 4) Live API-Tennis side
py -m src.data.fetch_upcoming_apitennis
py -m src.data.fetch_odds_apitennis
py -m src.data.fetch_player_roster
py -m src.data.fetch_player_photos

# 5) UI
py -m streamlit run streamlit_app.py
```

---

## API-Tennis configuration

The Upcoming, live ticker, head-to-head card, player photos and recent
results merge are all powered by [api-tennis.com](https://api-tennis.com).
To enable them locally:

```bash
cp .env.example .env
# then edit .env:
#   API_TENNIS_KEY=<your-key>
```

The `.env` file is in `.gitignore`. All API responses are cached on disk
under `data/cache/api_tennis/` to respect provider rate limits — TTL is
configurable via `API_TENNIS_CACHE_TTL_S` and
`API_TENNIS_ODDS_CACHE_TTL_S` in `.env`.

For the deployed app, set `API_TENNIS_KEY` as a GitHub repository secret
(Settings → Secrets and variables → Actions). The daily refresh workflow
uses it to pull live data automatically.

---

## Streamlit console

Six views, every player + tournament name is a click-through to its
profile:

1. **Matches Explorer** — every historical match scored with the AI's
   probability + a green/red row tint showing whether the AI was right.
   The "Confidence" column shows the market-aware blend by default.
2. **Upcoming** — live API-Tennis fixtures grouped by day → tournament,
   sorted by time. **LIVE** pill on matches currently in progress
   (red, animated). **FINAL** card with score + winner chip on matches
   that have finished today. Doubles + qualifying-tier matches are
   filtered out at fetch time.
3. **Players** — searchable + flag-tagged roster sorted active-first by
   win-rate. Profile cards include API-Tennis country flag, age, photo,
   Elo trajectory chart, surface breakdown, head-to-head jumps and an
   inlined "next match" card if the player has an upcoming fixture.
4. **Tournaments** — round-clustered cards (Final → Semifinals → R128),
   each match rendered with player photos, country flags and the final
   score.
5. **What-if** — pick any two players + surface + date and the model
   returns the probability split, with a head-to-head card and a
   **"Why this prediction" tale-of-the-tape** breakdown showing every
   relevant feature side-by-side.
6. **Leaderboard** — ranked cards (rank chip, photo, flag, full name,
   country) sortable by Win rate / Last 30d WR / AI accuracy on this
   player's matches / current streak / best surface.

A horizontal **live ticker** is pinned just under the top nav — every
match in progress scrolls past with player photos + the current set
score. Pauses on hover.

The top-right **ℹ️ About** dialog exposes data sources, the trained
model name + accuracy + log-loss, the current blend α, and the
no-leakage guarantee in plain language.

---

## Auto-refresh (GitHub Actions)

[`.github/workflows/daily-refresh.yml`](./.github/workflows/daily-refresh.yml)
runs every day at 04:00 UTC (manually triggerable from the Actions tab).
It:

1. Fetches the latest tennis-data.co.uk archive
2. Cleans + supplements with API-Tennis recent results (with the
   tour-level filter)
3. Rebuilds features, retrains the model, re-scores every match
4. Refreshes live fixtures, ATP roster + player photos
5. Commits the refreshed artifacts (predictions CSV, player metadata
   cache, model PKL, metrics.json, ~400 player photos per night) back
   to the repo so the deployed app stays fresh

`.gitignore` is written so the large source CSVs stay untracked but the
six artifacts the deployed app actually serves are tracked:

```
data/processed/all_predictions.csv
data/processed/fixtures_upcoming.csv
data/processed/recent_results_apitennis.csv
data/processed/matches_clean.csv
data/cache/player_meta.json
models/{logreg_final.pkl, imputer_final.pkl, feature_columns.txt, metrics.json}
```

---

## Architecture

```
┌──────────────────────────┐    ┌────────────────────────┐
│ tennis-data.co.uk        │    │ api-tennis.com         │
│ historical Excel files   │    │ live fixtures + odds + │
│ (2000 → today)           │    │ get_h2h + livescore +  │
│                          │    │ standings + photos     │
└────────────┬─────────────┘    └────────────┬───────────┘
             │                               │
             ▼                               ▼
┌──────────────────────────┐    ┌────────────────────────┐
│ src/data/                │    │ src/integrations/      │
│   fetch_data → CSV       │    │   api_tennis (cache +  │
│   preprocess             │    │   chunked windows)     │
│   cleaning               │    │ src/utils/             │
│   fetch_recent_results   │    │   player_meta          │
│   merge_recent_results   │    │   country (flag emoji) │
└────────────┬─────────────┘    │   surface inference    │
             ▼                  └────────────┬───────────┘
┌──────────────────────────┐                 │
│ src/features/            │                 │
│   elo (538-style MoV +   │                 │
│        dynamic K + tier) │                 │
│   form (+ momentum +     │                 │
│         weighted form)   │                 │
│   build_features         │                 │
│     ├─ H2H (+surface)    │                 │
│     ├─ common opponent   │                 │
│     ├─ tiebreak / set    │                 │
│     └─ tier / round      │                 │
└────────────┬─────────────┘                 │
             ▼                               │
┌──────────────────────────┐    ┌────────────────────────┐
│ src/models/              │    │ data/processed/        │
│   train_best (LR + HGB + │    │   all_predictions.csv  │
│   LightGBM + Ensemble +  │    │   fixtures_upcoming.csv│
│   calibration)           │    │   recent_results_*.csv │
│   score_all_matches      │    └────────────┬───────────┘
│     + per-year α blend   │                 │
└────────────┬─────────────┘                 │
             ▼                               ▼
┌────────────────────────────────────────────────────────┐
│                  streamlit_app.py                       │
│   nav · hero · KPIs · live ticker · About dialog        │
│   Matches · Upcoming · Players · Tournaments ·          │
│   What-if · Leaderboard                                 │
└────────────────────────────────────────────────────────┘
```

---

## Project layout

```
predictive-serve/
├─ data/
│  ├─ raw/             # tennis-data.co.uk merged Excel files (ignored)
│  ├─ processed/       # cleaned matches + feature matrix + predictions
│  └─ cache/           # API-Tennis cache + player_meta.json
├─ models/             # trained model, imputer, features, metrics.json
├─ assets/players/     # cached player photos (committed)
├─ src/
│  ├─ data/            # fetchers, preprocess, cleaning, merge, schema
│  ├─ features/        # elo, form, build_features
│  ├─ models/          # train_best, score_all_matches
│  ├─ integrations/    # API-Tennis client
│  ├─ predict/         # whatif single-match scoring
│  └─ utils/           # paths, env, feature_utils, surface, player_meta
├─ streamlit_app.py
├─ run_predictive_serve.bat
├─ requirements.txt
├─ test_quick.py
└─ .github/workflows/{ci,daily-refresh}.yml
```

---

## Development

```bash
# Smoke test (CI runs this on every push)
py test_quick.py

# Retrain + rescore manually
py -m src.models.train_best
py -m src.models.score_all_matches
```

### Roadmap (next iterations)

- Modularise `streamlit_app.py` (currently a single 3k+ line file) into
  per-tab modules under `src/ui/`.
- Migrate intermediate CSVs to Parquet for faster I/O and smaller disk.
- Surface SHAP-style feature importance in the About dialog.
- Kelly-criterion bet sizing when both an edge and a price are present.
- WTA support (the `get_standings` endpoint already takes `event_type=WTA`).
- Daily-digest push notification with the day's high-confidence picks.

Contributions welcome — open an issue to discuss bigger changes first.

---

## License

Intended for learning, portfolio and research use. Tennis match results
remain the property of their original sources (tennis-data.co.uk,
api-tennis.com).
