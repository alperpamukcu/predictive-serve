# src/models/score_all_matches.py
"""
Score every match in the training dataset with the persisted model.

Writes ``data/processed/all_predictions.csv`` with:
    date / surface / playerA / playerB / y      (meta)
    p_model    : raw, market-free AI probability that A wins
    pA_market  : bookmaker implied probability (when odds existed)
    edge       : p_model - pA_market
    p_blend    : alpha·p_model + (1-alpha)·pA_market when market exists, else p_model
                 alpha is fit on validation log-loss (see _fit_blend_alpha).
                 This closes ~half of the model-vs-market gap without
                 letting the market signal contaminate model training.

Also patches ``models/metrics.json`` with the chosen alpha and the val/test
log loss of p_blend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.feature_utils import load_feature_list


TRAIN_PATH = PROCESSED_DIR / "train_dataset.csv"
ALL_PRED_PATH = PROCESSED_DIR / "all_predictions.csv"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.txt"
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Same time-aware split used by train_best.py
TRAIN_END_YEAR = 2022
VAL_END_YEAR = 2025


def _fit_blend_alpha(p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> float:
    """Pick alpha ∈ [0, 1] minimising log loss of alpha·p_model + (1-alpha)·p_market on
    the rows we pass in. Uses a coarse grid + a fine grid around the best
    point — no scipy dependency required."""
    if len(y) < 200:
        return 1.0  # not enough data, trust the model
    grid_coarse = np.linspace(0.0, 1.0, 21)
    losses = []
    for a in grid_coarse:
        blend = np.clip(a * p_model + (1 - a) * p_market, 1e-7, 1 - 1e-7)
        losses.append(log_loss(y, blend))
    best_a = float(grid_coarse[int(np.argmin(losses))])
    # Refine
    lo = max(0.0, best_a - 0.05)
    hi = min(1.0, best_a + 0.05)
    grid_fine = np.linspace(lo, hi, 21)
    losses = []
    for a in grid_fine:
        blend = np.clip(a * p_model + (1 - a) * p_market, 1e-7, 1 - 1e-7)
        losses.append(log_loss(y, blend))
    return float(grid_fine[int(np.argmin(losses))])


def main() -> None:
    print(f"[score_all] Reading: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    feature_cols = load_feature_list(FEATURE_LIST_PATH)
    print(f"[score_all] Feature count: {len(feature_cols)}")

    y0 = pd.to_numeric(df.get("y"), errors="coerce")
    ok = y0.notna()
    if not bool(ok.all()):
        df = df.loc[ok].copy()
    y = df["y"].astype(int).values

    X = df[feature_cols].copy()
    print("[score_all] Loading model + imputer...")
    model = load(MODEL_PATH)
    imputer = load(IMPUTER_PATH)
    X_imp = imputer.transform(X)
    p_model = model.predict_proba(X_imp)[:, 1]

    # Market and edge
    if "pA_market" in df.columns:
        p_market = pd.to_numeric(df["pA_market"], errors="coerce").astype(float).values
    else:
        p_market = np.full_like(p_model, np.nan, dtype=float)
    edge = p_model - p_market

    # Fit alpha on validation rows where market exists (2022-2024).
    yr = df["date"].dt.year.values
    val_mask = (yr >= TRAIN_END_YEAR) & (yr < VAL_END_YEAR) & np.isfinite(p_market)
    if val_mask.sum() >= 500:
        alpha = _fit_blend_alpha(p_model[val_mask], p_market[val_mask], y[val_mask])
    else:
        alpha = 1.0
    print(f"[score_all] Market-prior blend alpha (global) = {alpha:.3f} "
          f"({int(val_mask.sum())} val rows used)")

    # P1.3: per-year alpha so the blend tracks market drift across seasons.
    # Fit alpha on each (2022, 2023, 2024) slice; for years without enough
    # samples fall back to the global alpha.
    per_year_alpha: Dict[int, float] = {}
    for year in (TRAIN_END_YEAR, TRAIN_END_YEAR + 1, TRAIN_END_YEAR + 2):
        mask = (yr == year) & np.isfinite(p_market)
        if mask.sum() >= 500:
            per_year_alpha[year] = _fit_blend_alpha(p_model[mask], p_market[mask], y[mask])
        else:
            per_year_alpha[year] = alpha
    # Test seasons (2025+) and live predictions use the most recent val
    # year's alpha — that's the best-calibrated coefficient we have.
    latest_val_year = max(per_year_alpha)
    print(f"[score_all] per-year alphas: {per_year_alpha} "
          f"(2025+/live use {latest_val_year}'s alpha = {per_year_alpha[latest_val_year]:.3f})")

    def _alpha_for(year: int) -> float:
        if year in per_year_alpha:
            return per_year_alpha[year]
        return per_year_alpha[latest_val_year]

    alphas = np.array([_alpha_for(int(y_)) for y_ in yr])
    p_blend_full = np.where(
        np.isfinite(p_market),
        alphas * p_model + (1 - alphas) * p_market,
        p_model,
    )
    p_blend_full = np.clip(p_blend_full, 1e-7, 1 - 1e-7)

    meta_keep = ["date", "surface", "playerA", "playerB", "y"]
    for extra in ("tournament", "round"):
        if extra in df.columns:
            meta_keep.append(extra)
    out = df[meta_keep].copy()
    out["p_model"] = p_model
    out["pA_market"] = p_market
    out["edge"] = edge
    out["p_blend"] = p_blend_full

    ALL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ALL_PRED_PATH, index=False)
    print(f"[score_all] Saved {len(out):,} rows -> {ALL_PRED_PATH}")

    # Report val/test metrics for both p_model and p_blend
    def _report(mask: np.ndarray, label: str):
        if mask.sum() == 0:
            return None
        y_ = y[mask]
        pm = p_model[mask]
        pb = p_blend_full[mask]
        return {
            "n": int(mask.sum()),
            "model": {
                "logloss": float(log_loss(y_, pm)),
                "brier": float(brier_score_loss(y_, pm)),
                "accuracy": float(accuracy_score(y_, (pm >= 0.5).astype(int))),
            },
            "blend": {
                "logloss": float(log_loss(y_, pb)),
                "brier": float(brier_score_loss(y_, pb)),
                "accuracy": float(accuracy_score(y_, (pb >= 0.5).astype(int))),
            },
        }

    val_report = _report(val_mask, "validation")
    test_mask = (yr >= VAL_END_YEAR) & np.isfinite(p_market)
    test_report = _report(test_mask, "test")

    # Patch metrics.json (created by train_best)
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    metrics["blend"] = {
        "alpha": alpha,
        "per_year_alpha": per_year_alpha,
        "val": val_report,
        "test": test_report,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[score_all] Patched {METRICS_PATH} with blend stats.")
    if val_report:
        print(
            "[score_all] Val (market-only): "
            f"model logloss {val_report['model']['logloss']:.4f} -> "
            f"blend logloss {val_report['blend']['logloss']:.4f}  "
            f"(acc {val_report['model']['accuracy']*100:.2f}% -> "
            f"{val_report['blend']['accuracy']*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
