# Model improvement roadmap

The current production model is a calibrated **HistGradientBoostingClassifier**
trained on 36 leakage-safe features. As of the last refresh it scores:

| Split      | Log loss | Accuracy |
|------------|----------|----------|
| Validation | 0.610    | 65.5%    |
| Held-out 2025 test | ~0.62 | ~65% |
| Market baseline (val) | 0.582 | 68.1% |

The gap to the bookmaker baseline (~3 percentage points of accuracy) is
the headroom we're trying to close. Below is a prioritised list of
experiments, each scoped so it can ship as a standalone PR.

---

## Phase 1 — Richer feature engineering (1–2 wins ~1 pt each)

### 1.1 Surface-aware H2H
**What**: split `h2h_winrateA`/`h2h_winrateB` by surface. Hard-court Federer
vs Nadal is a different signal than clay Federer vs Nadal.
**Where**: `src/features/build_features.py::add_h2h_features` — store
three sets of counters keyed by `(playerA, playerB, surface)`.
**Expected**: +0.2 to +0.5 pp accuracy on slams.

### 1.2 Recent-form *trajectory* (not just rate)
**What**: in addition to last-5/last-10 win rate, add the slope of the
running Elo over those windows. A player on the rise (positive slope)
typically beats a player at the same Elo who is plateauing.
**Where**: new `src/features/trajectory.py` that diffs the rolling Elo
window.
**Expected**: +0.3 pp.

### 1.3 Tournament-level Elo
**What**: separate Elo per tournament tier (slam, masters, 250). Many
top players coast through 250s and ramp up at slams.
**Where**: extend `src/features/elo.py` with a `tier_elo` dictionary.
**Expected**: +0.2 pp, larger lift on tier-mismatched matchups.

### 1.4 Days-since-last-tournament-win
**What**: morale / momentum feature — number of days since the player
last won a title. Replaces a noisy "days since last match" with a more
meaningful signal.
**Where**: `src/features/form.py`.
**Expected**: +0.1 pp on its own, helpful for explainability.

### 1.5 Serve / return stats from API-Tennis match payload
**What**: API-Tennis `get_fixtures` includes a `statistics` array per
match (1st serve %, aces, double faults, break-points won). Aggregate
per player over a rolling window and add as features.
**Where**: new `src/features/serve_stats.py`, fed by a new
`fetch_match_stats_apitennis.py`.
**Expected**: +0.5–1.0 pp. This is the biggest single-feature win because
we currently have no serve/return signal.

---

## Phase 2 — Better model choices (1 win ~0.5–1 pt)

### 2.1 LightGBM
**What**: drop-in replacement for `HistGradientBoostingClassifier`.
Faster, slightly better defaults, native categorical support so we can
encode surface / round / tournament-tier without one-hot blow-up.
**Where**: add `lightgbm` to `requirements.txt`, swap in
`src/models/train_best.py`.
**Expected**: +0.3–0.5 pp, 2–3× faster training.

### 2.2 Probability stacking
**What**: train two specialists (HGB no-market + LR with form-only) and
stack them with a logistic meta-model. Stacking often picks up the last
~0.3 pp the base learners leave on the table.
**Where**: `src/models/train_stacked.py`.

### 2.3 Per-surface models
**What**: train three separate models (hard / clay / grass) and route at
inference time. Surface-specific feature importances differ — for
example serve dominance matters more on grass.
**Where**: `src/models/train_per_surface.py`, expose via a thin router
in `src/predict/whatif.py`.
**Expected**: +0.5 pp on the smaller surfaces (grass especially).

### 2.4 Better calibration
**What**: replace Platt/Isotonic with a **temperature-scaling** post-hoc
fit on the 2021 calibration slice. Temperature scaling preserves the
ordering but shrinks confident probabilities slightly, which usually
helps log-loss without hurting accuracy.
**Where**: extend `src/models/train_best.py::_calibrate`.

---

## Phase 3 — Bigger structural changes (highest ceiling, longest lead)

### 3.1 Use bookmaker odds as a *prior* (not a feature)
**What**: keep the "pure" AI model market-free, but at inference time
blend `α · p_model + (1-α) · p_market` where α is fit on validation.
This is the cheapest way to close the gap to the bookmaker baseline and
still gives a meaningful "edge" via the pure model.
**Where**: `src/predict/whatif.py` + UI toggle.

### 3.2 In-match features
**What**: when a match is currently live, the API gives us set scores,
serve %, break-points. A separate "in-match" model that updates the win
probability after each game opens up a whole new use-case (live trading
companion).

### 3.3 WTA expansion
**What**: re-run the entire pipeline against tennis-data.co.uk's WTA
archive. Almost everything generalises directly; only a small number of
player-name normalisation tweaks are needed.

---

## Suggested execution order

1. **1.1 Surface H2H**, **1.2 Elo trajectory**, **1.4 Title days** —
   all small, additive, and easy to verify against val log-loss.
2. **2.1 LightGBM** — single drop-in switch, likely the cleanest +0.5pp.
3. **3.1 Market prior** — non-intrusive, gives users an immediate
   accuracy boost while we work on the harder feature work.
4. **1.5 Serve stats** — biggest individual lift, but needs new API
   ingestion + storage; do it once the simpler wins are banked.
5. **2.3 Per-surface models** — only after 1.x features are in place,
   because per-surface models are also more sensitive to feature noise.

Each experiment must:
- write `models/metrics.json` so improvements are tracked across
  commits;
- be runnable from the daily-refresh workflow;
- keep `LEAKY_MARKET_COLS` excluded unless the experiment is the
  explicit market-aware variant in §3.1.
