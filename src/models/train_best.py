"""
Production trainer for Predictive Serve.

Selects the best leakage-safe model (LogReg vs HistGradientBoosting), applies
a calibration pass, evaluates on a held-out test split, and persists artifacts
that downstream scripts (score_all_matches, streamlit_app, whatif) can load.

Key rules:
- Market-derived columns are NEVER used as model inputs. The "edge" between
  model probability and market probability stays meaningful.
- Time-aware splits: train < 2022, validation 2022-2024, test >= 2025.
- Calibration uses a within-train slice (year == 2021) so val/test never
  influence the calibrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.feature_utils import select_model_features

try:
    import lightgbm as lgb  # type: ignore
    _LGB_AVAILABLE = True
except Exception:
    _LGB_AVAILABLE = False

TRAIN_DATA_PATH = PROCESSED_DIR / "train_dataset.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

TRAIN_END_YEAR = 2022   # train: year < 2022
VAL_END_YEAR = 2025     # val: 2022..2024 ;  test: >= 2025
CALIB_YEAR = 2021       # within-train slice used as calibration set


@dataclass
class Metrics:
    n: int
    logloss: float
    brier: float
    accuracy: float

    def asdict(self) -> dict:
        return asdict(self)


def _eval(y: np.ndarray, p: np.ndarray) -> Metrics:
    return Metrics(
        n=int(len(y)),
        logloss=float(log_loss(y, p)),
        brier=float(brier_score_loss(y, p)),
        accuracy=float(accuracy_score(y, (p >= 0.5).astype(int))),
    )


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    yr = df["date"].dt.year
    train = df[yr < TRAIN_END_YEAR].copy()
    val = df[(yr >= TRAIN_END_YEAR) & (yr < VAL_END_YEAR)].copy()
    test = df[yr >= VAL_END_YEAR].copy()
    return train, val, test


def _grid_hgb() -> List[HistGradientBoostingClassifier]:
    """Expanded grid (~20 configs) with early stopping enabled — Phase 2.3.
    Sweeps learning rate × depth × leaves × min-samples × L2 reg. Early
    stopping prevents the wider configs from over-training."""
    grid = []
    configs = [
        # (lr, max_depth, max_leaf_nodes, min_samples_leaf, l2, max_iter)
        (0.05, 6, 31, 50, 0.0, 600),
        (0.05, 4, 31, 50, 0.0, 600),
        (0.03, 6, 31, 50, 0.0, 800),
        (0.03, 4, 31, 50, 0.0, 800),
        (0.02, 6, 31, 50, 0.0, 1200),
        (0.05, 6, 63, 50, 0.0, 600),
        (0.03, 6, 63, 50, 0.0, 800),
        (0.05, 8, 63, 80, 0.0, 600),
        (0.03, 8, 63, 80, 0.0, 800),
        (0.05, 6, 31, 100, 0.1, 600),
        (0.03, 6, 31, 100, 0.1, 800),
        (0.05, 6, 63, 100, 0.1, 600),
        (0.03, 6, 63, 100, 0.5, 800),
        (0.05, 4, 31, 200, 0.5, 600),
        (0.03, 6, 31, 200, 1.0, 800),
        (0.02, 6, 63, 50, 0.0, 1200),
        (0.02, 4, 31, 100, 0.1, 1200),
        (0.07, 5, 31, 50, 0.0, 500),
        (0.07, 6, 63, 80, 0.1, 500),
        (0.04, 6, 47, 50, 0.05, 700),
    ]
    for lr_, md_, mln_, msl_, l2_, it_ in configs:
        grid.append(
            HistGradientBoostingClassifier(
                learning_rate=lr_,
                max_depth=md_,
                max_leaf_nodes=mln_,
                min_samples_leaf=msl_,
                l2_regularization=l2_,
                max_iter=it_,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=42,
            )
        )
    return grid


def _calibrate(model, imputer: SimpleImputer, feats: List[str], df: pd.DataFrame) -> Optional[CalibratedClassifierCV]:
    """
    Calibrate ``model`` using only within-train (year == CALIB_YEAR) data.
    The model is refit on year < CALIB_YEAR, then frozen-calibrated on the
    held-out within-train year. Returns None when slices are too small.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    yr = df["date"].dt.year

    pre = df[yr < CALIB_YEAR]
    cal = df[yr == CALIB_YEAR]
    if len(pre) < 10_000 or len(cal) < 1_500:
        return None

    X_pre = imputer.fit_transform(pre[feats])
    y_pre = pre["y"].astype(int).values
    X_cal = imputer.transform(cal[feats])
    y_cal = cal["y"].astype(int).values

    # Refit base on the calibration train slice
    model.fit(X_pre, y_pre)

    try:
        from sklearn.frozen import FrozenEstimator  # type: ignore
        prefit_est = FrozenEstimator(model)
        cv = None
    except Exception:
        prefit_est = model
        cv = "prefit"

    best = None
    best_ll = float("inf")
    for method in ("sigmoid", "isotonic"):
        try:
            cal_clf = CalibratedClassifierCV(prefit_est, method=method, cv=cv)  # type: ignore[arg-type]
            cal_clf.fit(X_cal, y_cal)
            p = cal_clf.predict_proba(X_cal)[:, 1]
            ll = float(log_loss(y_cal, p))
            if ll < best_ll:
                best_ll = ll
                best = cal_clf
        except Exception as e:
            print(f"[best] calibration({method}) skipped: {e}")
    return best


def train_and_select() -> Tuple[Path, Path, Path]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_DATA_PATH}")

    print(f"[best] Reading: {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    train_df, val_df, test_df = _split(df)
    print(f"[best] Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    feats = select_model_features(list(df.columns), include_market=False)
    print(f"[best] Feature count (no market): {len(feats)}")

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train_df[feats])
    X_va = imp.transform(val_df[feats])
    y_tr = train_df["y"].astype(int).values
    y_va = val_df["y"].astype(int).values

    candidates = []

    # 1) Logistic regression baseline
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, n_jobs=-1, solver="lbfgs"),
    )
    lr.fit(X_tr, y_tr)
    m_lr = _eval(y_va, lr.predict_proba(X_va)[:, 1])
    print(f"[best] LR val={m_lr.asdict()}")
    candidates.append(("LogReg", lr, m_lr))

    # 2) Hist gradient boosting grid
    best_hgb = None
    best_hgb_metrics: Optional[Metrics] = None
    for hgb in _grid_hgb():
        hgb.fit(X_tr, y_tr)
        m = _eval(y_va, hgb.predict_proba(X_va)[:, 1])
        if best_hgb_metrics is None or m.logloss < best_hgb_metrics.logloss:
            best_hgb = hgb
            best_hgb_metrics = m
    assert best_hgb is not None and best_hgb_metrics is not None
    print(f"[best] HGB val={best_hgb_metrics.asdict()}")
    candidates.append(("HGB", best_hgb, best_hgb_metrics))

    # 3) LightGBM (Phase 2.1). A small grid; usually +0.3-0.5pp over HGB
    # and 2-3x faster training.
    if _LGB_AVAILABLE:
        lgb_grid = [
            dict(n_estimators=600, learning_rate=0.05, num_leaves=31,  max_depth=-1, min_child_samples=50,  reg_lambda=0.0),
            dict(n_estimators=800, learning_rate=0.03, num_leaves=31,  max_depth=-1, min_child_samples=50,  reg_lambda=0.0),
            dict(n_estimators=600, learning_rate=0.05, num_leaves=63,  max_depth=-1, min_child_samples=50,  reg_lambda=0.1),
            dict(n_estimators=800, learning_rate=0.03, num_leaves=63,  max_depth=-1, min_child_samples=100, reg_lambda=0.1),
            dict(n_estimators=800, learning_rate=0.03, num_leaves=127, max_depth=-1, min_child_samples=100, reg_lambda=0.1),
            dict(n_estimators=1000,learning_rate=0.02, num_leaves=31,  max_depth=-1, min_child_samples=50,  reg_lambda=0.0),
            dict(n_estimators=1000,learning_rate=0.02, num_leaves=63,  max_depth=8,  min_child_samples=80,  reg_lambda=0.2),
            dict(n_estimators=600, learning_rate=0.07, num_leaves=31,  max_depth=-1, min_child_samples=50,  reg_lambda=0.0),
        ]
        best_lgb = None
        best_lgb_metrics: Optional[Metrics] = None
        for params in lgb_grid:
            mdl = lgb.LGBMClassifier(
                **params, n_jobs=-1, random_state=42, verbose=-1
            )
            mdl.fit(X_tr, y_tr)
            m = _eval(y_va, mdl.predict_proba(X_va)[:, 1])
            if best_lgb_metrics is None or m.logloss < best_lgb_metrics.logloss:
                best_lgb = mdl
                best_lgb_metrics = m
        if best_lgb is not None and best_lgb_metrics is not None:
            print(f"[best] LightGBM val={best_lgb_metrics.asdict()}")
            candidates.append(("LightGBM", best_lgb, best_lgb_metrics))
    else:
        best_lgb = None
        print("[best] LightGBM not installed — skipping that candidate.")

    # 4) Soft-voting ensemble (Phase 2.2). Averaging diverse learners — a
    # linear model + two gradient-boosters — usually shaves a little more
    # log-loss than any single one because their errors decorrelate.
    estimators = [("lr", lr), ("hgb", best_hgb)]
    if best_lgb is not None:
        estimators.append(("lgb", best_lgb))
    try:
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        ensemble.fit(X_tr, y_tr)
        m_ens = _eval(y_va, ensemble.predict_proba(X_va)[:, 1])
        print(f"[best] Ensemble val={m_ens.asdict()}")
        candidates.append(("Ensemble", ensemble, m_ens))
    except Exception as e:
        print(f"[best] Ensemble skipped: {e}")

    # Pick the best on validation log-loss
    best_name, best_model, best_metrics = min(candidates, key=lambda c: c[2].logloss)
    print(f"[best] Selected base: {best_name} val_logloss={best_metrics.logloss:.6f}")

    # Calibration pass
    cal_imp = SimpleImputer(strategy="median")
    cal_model = _calibrate(best_model, cal_imp, feats, df)
    if cal_model is not None:
        X_va_c = cal_imp.transform(val_df[feats])
        m_cal = _eval(y_va, cal_model.predict_proba(X_va_c)[:, 1])
        print(f"[best] Calibrated val={m_cal.asdict()}")
        if m_cal.logloss < best_metrics.logloss:
            best_model = cal_model
            best_name = f"{best_name}+cal"
            best_metrics = m_cal
            imp = cal_imp  # the calibrated path uses its own imputer fit

    # Held-out test evaluation (never tuned against)
    test_metrics_dict = None
    if len(test_df) > 0:
        X_te = imp.transform(test_df[feats])
        y_te = test_df["y"].astype(int).values
        m_te = _eval(y_te, best_model.predict_proba(X_te)[:, 1])
        test_metrics_dict = m_te.asdict()
        print(f"[best] {best_name} TEST={test_metrics_dict}")

    # Market baseline (for reporting, on val rows where odds exist)
    market_baseline = None
    if "pA_market" in val_df.columns and "has_market" in val_df.columns:
        mask = val_df["has_market"].astype(int) == 1
        if mask.sum() > 1000:
            mb = _eval(
                val_df.loc[mask, "y"].astype(int).values,
                pd.to_numeric(val_df.loc[mask, "pA_market"], errors="coerce").astype(float).values,
            )
            market_baseline = mb.asdict()
            print(f"[best] Market baseline (val) {market_baseline}")

    # Persist artifacts (filenames preserved for backwards compatibility)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "logreg_final.pkl"
    imputer_path = MODELS_DIR / "imputer_final.pkl"
    feature_cols_path = MODELS_DIR / "feature_columns.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(imp, imputer_path)
    feature_cols_path.write_text("\n".join(feats) + "\n", encoding="utf-8")

    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": best_name,
                "n_features": len(feats),
                "validation": best_metrics.asdict(),
                "test": test_metrics_dict,
                "market_baseline_val": market_baseline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[best] Saved model -> {model_path}")
    print(f"[best] Saved imputer -> {imputer_path}")
    print(f"[best] Saved features -> {feature_cols_path} (n={len(feats)})")
    print(f"[best] Metrics -> {METRICS_PATH}")
    return model_path, imputer_path, feature_cols_path


if __name__ == "__main__":
    train_and_select()
