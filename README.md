## Predictive Serve 🎾

**Predictive Serve** is an end-to-end Python project for **tennis match forecasting** with a reproducible pipeline, Sportradar integrations (upcoming schedule + odds), and a polished Streamlit UI.

### What it does
- **Historical pipeline**: downloads and processes ATP match data from `tennis-data.co.uk` (seasons **2000–2026**)
- **Feature engineering**: Elo, form, workload, head-to-head, surface, and optional market-derived features
- **Model**: Logistic Regression baseline (fast, debuggable)
- **Upcoming matches**: pulls fixtures from **Sportradar Tennis API** and enriches with **Sportradar Odds Comparison (OC Regular)** consensus lines (when available)
- **UI**: Streamlit app (Matches Explorer, Upcoming, What-if, Players, Tournaments, Leaderboards)

This repository is intended for **learning / portfolio / research** use. If you deploy it publicly, read the security & operations notes below.

---

## 1. Quick Start

### 1.1. Prerequisites

- **OS**: Windows 10/11 (for the `.bat` convenience script) – other OSes can run the Python commands manually.
- **Python**: 3.10+  
  On Windows, the project expects the **`py` launcher** to exist (`py --version` should work).

Clone the repo:

```bash
git clone https://github.com/your-user/predictive-serve.git
cd predictive-serve
```

### 1.2. Easiest way (Windows one‑click)

From File Explorer, in the project root:

1. Double‑click **`run_predictive_serve.bat`**
2. Wait while it:
   - Installs Python dependencies from `requirements.txt`
   - Downloads match data for seasons **2000–2026**
   - Runs the full data & feature pipeline
   - Trains the Logistic Regression model
   - Scores all matches and writes `data/processed/all_predictions.csv`
   - (Optional) fetches upcoming fixtures + odds via Sportradar
   - Starts the Streamlit app
3. When it finishes, your browser will open (or you can go to `http://localhost:8501`).

You can also run it from a terminal:

```bash
run_predictive_serve.bat
```

If anything fails, the script stops with a clear `[ERROR] ...` or `[WARN] ...` message.

### 1.3. Manual setup (any OS)

Create (optionally) and activate a virtual env, then:

```bash
pip install -r requirements.txt
```

Run the pipeline manually:

```bash
# 1) Fetch raw data from tennis-data.co.uk (seasons 2000–2026)
py -m src.data.fetch_data

# 2) Preprocess + cleaning
py -m src.data.preprocess
py -m src.data.cleaning

# 3) Feature engineering
py -m src.features.elo
py -m src.features.form
py -m src.features.sets      # set-based features (best-of-3/5), best-effort
py -m src.features.build_features

# 4) Train model and score all matches
py -m src.models.train_logreg
py -m src.models.score_all_matches
```

Then start the UI:

```bash
py -m streamlit run streamlit_app.py
```

---

## 2. Sportradar (Upcoming + Odds + Assets)

### 2.1. Configure environment variables

Copy `.env.example` to `.env` and fill the values (never commit `.env`):

```bash
cp .env.example .env
```

Required for **Upcoming fixtures**:
- `SPORTRADAR_TENNIS_API_KEY`

Optional (adds value):
- `SPORTRADAR_ODDS_API_KEY` (Odds Comparison Regular / OC Regular)
- `SPORTRADAR_IMAGES_API_KEY` (Images API v3 — may be restricted, see below)

### 2.2. Fetch upcoming fixtures (Sportradar Tennis)

```bash
py -m src.data.fetch_upcoming_sportradar
```

Writes: `data/processed/fixtures_upcoming.csv`

### 2.3. Enrich fixtures with odds (OC Regular)

```bash
py -m src.data.fetch_odds_sportradar
```

Notes:
- Odds availability depends on your OC package/coverage. Not all fixtures will have odds.
- Trial keys may hit **429 rate limits**; the fetcher implements backoff and skips days safely.

### 2.4. Player images / tournament logos (Images API v3)

```bash
py -m src.data.fetch_player_images_sportradar
py -m src.data.fetch_tournament_logos_sportradar
```

If your key or plan is not entitled for Images API, you may see **403 Authentication Error**.
The UI is still usable thanks to a built-in offline **avatar placeholder** fallback.

---

## 3. Project Overview

### 3.1. Problem

We want to answer:

> *“Given tennis players’ form, Elo rating, head‑to‑head history, rest, tournament and surface information – how good can we get at predicting match outcomes compared to betting markets?”*

Key elements:
- **Data source**: `tennis-data.co.uk` ATP historical match files
- **Target**: `y` (1 = player A wins, 0 = player B wins)
- **Model output**: \( P(\text{player A wins}) \)
- **Baseline**: implied probabilities from bookmaker odds

### 3.2. High-level pipeline

```text
1. src/data/fetch_data.py
   → data/raw/allyears.csv        (2000–2026)

2. src/data/preprocess.py
   → data/processed/matches_allyears.csv

3. src/data/cleaning.py
   → data/processed/matches_clean.csv

4. src/features/elo.py
   → data/processed/matches_with_elo.csv

5. src/features/form.py
   → data/processed/matches_with_elo_form.csv

6. src/features/sets.py
   → data/processed/matches_with_elo_form_sets.csv

7. src/features/build_features.py
   → data/processed/train_dataset.csv

8. src/models/train_logreg.py
   → models/logreg_final.pkl
   → models/imputer_final.pkl
   → models/feature_columns.txt

9. src/models/score_all_matches.py
   → data/processed/all_predictions.csv

10. streamlit_app.py / src/predict/whatif.py
    → interactive UI / CLI predictions
```

---

## 4. Directory Structure

```text
predictive-serve/
├─ data/
│  ├─ raw/
│  │  └─ allyears.csv                  # merged raw tennis-data.co.uk seasons
│  └─ processed/
│     ├─ matches_allyears.csv         # normalized from allyears
│     ├─ matches_clean.csv            # cleaned subset
│     ├─ matches_with_elo.csv         # Elo features added
│     ├─ matches_with_elo_form.csv    # form and workload features
│     ├─ matches_with_elo_form_sets.csv
│     ├─ train_dataset.csv            # final training feature matrix
│     ├─ all_predictions.csv          # model scores for all matches
│     └─ val_predictions.csv          # (optional) validation predictions
│
├─ models/
│  ├─ logreg_final.pkl                # Logistic Regression pipeline
│  ├─ imputer_final.pkl               # SimpleImputer for missing values
│  └─ feature_columns.txt             # ordered list of feature column names
│
├─ notebooks/
│  ├─ 01_eda_matches_allyears.ipynb   # exploratory data analysis
│  └─ 02_train_models.ipynb           # model comparison / metrics
│
├─ src/
│  ├─ data/
│  │  ├─ fetch_data.py                # download Excel files and merge → allyears.csv
│  │  ├─ preprocess.py                # normalize raw data → matches_allyears.csv
│  │  ├─ cleaning.py                  # filters / sanity checks → matches_clean.csv
│  │  └─ schema.py                    # canonical column definitions
│  │
│  ├─ features/
│  │  ├─ elo.py                       # global + surface Elo ratings
│  │  ├─ form.py                      # short‑term form and workload
│  │  ├─ sets.py                      # set‑level performance (best effort)
│  │  └─ build_features.py            # combine all features → train_dataset.csv
│  │
│  ├─ models/
│  │  ├─ train_logreg.py              # train Logistic Regression
│  │  └─ score_all_matches.py         # score all historical matches
│  │
│  ├─ predict/
│  │  └─ whatif.py                    # scenario‑based single‑match predictions
│  │
│  ├─ analysis/
│  │  └─ metrics.py                   # model vs market performance metrics
│  │
│  └─ utils/
│     ├─ config.py                    # shared paths (PROJECT_ROOT, DATA_DIR, etc.)
│     └─ feature_utils.py             # loading saved feature lists
│
├─ streamlit_app.py                   # main Streamlit UI
├─ test_quick.py                      # lightweight sanity checks
├─ test_system.py                     # full end‑to‑end system tests
├─ run_predictive_serve.bat           # Windows one‑click launcher
├─ requirements.txt                   # Python dependencies
└─ .gitignore
```

---

## 5. Components in More Detail

### 4.1. Data layer (`src/data`)

- **`config.py` (in `src/utils`)**  
  Central place for paths:
  - `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`
  - `MODELS_DIR`, `NOTEBOOKS_DIR`
  - `ALLYEARS_PATH` (canonical path for `allyears.csv`)

- **`schema.py`**  
  Defines the canonical match schema (`MATCH_COLUMNS`), including:
  - `date`, `tourney`, `surface`, `round`
  - `playerA`, `playerB`, `rankA`, `rankB`
  - `oddsA`, `oddsB`, `winner`
  - `playerA_norm`, `playerB_norm`

- **`fetch_data.py`**  
  - Downloads Excel files from `tennis-data.co.uk` for seasons **2000–2026**
  - Merges them into a single CSV: `data/raw/allyears.csv`
  - Handles missing seasons gracefully (older `.xls` years require `xlrd` if you want them)

- **`preprocess.py`**  
  - Normalizes column names and types
  - Parses dates
  - Normalizes player names (lowercase / trimmed)
  - Computes fair implied probabilities from odds
  - Writes `data/processed/matches_allyears.csv`

- **`cleaning.py`**  
  - Drops rows with missing critical fields
  - Filters to matches between **2000 and 2026**
  - Ensures odds are numeric and within a reasonable range
  - Removes unfinished matches (retired, walkover, etc.)
  - Writes `data/processed/matches_clean.csv`

### 4.2. Feature engineering (`src/features`)

- **`elo.py`**  
  - Computes **global Elo** and **surface‑specific Elo** ratings (Hard, Clay, Grass, etc.)
  - Adds `eloA`, `eloB`, `elo_diff`, `elo_surfaceA`, `elo_surfaceB`, `elo_surface_diff`
  - Saves to `matches_with_elo.csv`

- **`form.py`**  
  - Computes short‑term form and workload:
    - Win rate over the last 5 and 10 matches
    - Days since last match (clipped)
    - Number of matches in the last 30 days
  - Saves to `matches_with_elo_form.csv`

- **`sets.py`**  
  - Best‑effort set‑level stats (overall, best‑of‑3, best‑of‑5) where scoring data is available
  - Saves to `matches_with_elo_form_sets.csv`

- **`build_features.py`**  
  - Adds:
    - Head‑to‑head features (`h2h_matches`, win rates)
    - Round importance and tournament tier
    - Surface one‑hot features
    - Market features (`pA_market`, `pB_market`, `p_diff`, `logit_pA_market`)
    - Various “A vs B” differences (Elo diff, form diff, rank diff, etc.)
  - Outputs `data/processed/train_dataset.csv` with:
    - Meta columns: `date`, `surface`, `playerA`, `playerB`, `y`
    - ~35–40 feature columns used by the model

### 4.3. Modeling (`src/models`)

- **`train_logreg.py`**
  - Reads `train_dataset.csv`
  - Splits into train / validation by date (pre‑2022 vs 2022+)
  - Builds a pipeline: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`
  - Excludes raw market odds features from training to avoid leakage
  - Saves:
    - `models/logreg_final.pkl`
    - `models/imputer_final.pkl`
    - `models/feature_columns.txt`
  - Prints validation metrics (log‑loss, Brier, accuracy)

- **`score_all_matches.py`**
  - Loads the model, imputer, and feature list
  - Applies them to `train_dataset.csv`
  - Writes `data/processed/all_predictions.csv` with:
    - `date`, `surface`, `playerA`, `playerB`, `y`
    - `p_model` (model probability A wins)
    - `pA_market` and `edge = p_model - pA_market` when odds exist

### 4.4. UI and predictions

- **`streamlit_app.py`**
  - Main web UI with three main views:
    1. **Matches** – filter matches, compare model vs market, inspect edges
    2. **What‑if** – scenario‑based prediction for arbitrary matchups
    3. **Leaderboard** – player ranking and performance aggregates
  - Uses `all_predictions.csv` plus raw/feature data when needed.

- **`src/predict/whatif.py`**
  - CLI and internal API for “what‑if” scenarios:
    - Build a single feature row for a hypothetical match
    - Run it through the same model and imputer as the main pipeline
  - Example CLI usage:

    ```bash
    py -m src.predict.whatif --playerA "Roger Federer" --playerB "Rafael Nadal" --surface "Hard" --date "2020-01-15"
    ```

---

## 6. Testing

### 5.1. Quick structural test

```bash
py test_quick.py
```

Checks that the basic folder/file structure and imports look correct.

### 5.2. Full system test

```bash
py test_system.py
```

Runs a more complete end‑to‑end test including:
- Data paths and existence of key files
- Ability to import key modules
- Model loading
- Basic pipeline sanity checks

---

## 7. Troubleshooting

- **`ModuleNotFoundError` or missing packages**
  - Run:
    ```bash
    pip install -r requirements.txt
    ```

- **`FileNotFoundError: train_dataset.csv` or similar**
  - Re‑run the pipeline:
    ```bash
    py -m src.data.fetch_data
    py -m src.data.preprocess
    py -m src.data.cleaning
    py -m src.features.elo
    py -m src.features.form
    py -m src.features.sets
    py -m src.features.build_features
    py -m src.models.train_logreg
    py -m src.models.score_all_matches
    ```

- **Streamlit fails to start**
  - Confirm:
    ```bash
    py -m pip install streamlit
    py -m streamlit run streamlit_app.py
    ```

---

## 8. Security & Operations (for public deployments)

If you deploy this publicly, do **not** expose keys to clients.

- **Keys**: keep API keys in server-side environment variables (Streamlit Cloud secrets / Docker secrets / GitHub Actions secrets).
- **No client-side calls**: Sportradar calls must be made from the backend (this repo does that).
- **Refresh strategy**:
  - Prefer a server-side scheduler (cron/GitHub Actions) that runs:
    - `py -m src.data.fetch_upcoming_sportradar`
    - `py -m src.data.fetch_odds_sportradar`
  - Then the UI reads fresh CSVs. This keeps keys out of the browser and reduces rate-limit risk.
- **Rate limits**: trial keys can receive **429**. The fetchers back off and continue safely.
- **Data artifacts**: generated datasets and model binaries can be large. In this repo they are ignored by default via `.gitignore`. For releases, keep only small examples in `data/examples/`.


- All important paths are centralized in `src/utils/config.py`.
- Feature names used by the model are stored in `models/feature_columns.txt`.
- The current pipeline uses seasons **2000–2026** from `tennis-data.co.uk`.  
  Earlier `.xls` seasons may require installing `xlrd` if you want to extend coverage.

