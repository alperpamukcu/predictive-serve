from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.utils.config import PROCESSED_DIR


MODELS_DIR = PROCESSED_DIR.parent.parent / "models"
TRAIN_DATA_PATH = PROCESSED_DIR / "train_dataset.csv"


@dataclass(frozen=True)
class Metrics:
    logloss: float
    brier: float
    acc: float
    n: int


def _eval(y: np.ndarray, p: np.ndarray) -> Metrics:
    return Metrics(
        logloss=float(log_loss(y, p)),
        brier=float(brier_score_loss(y, p)),
        acc=float(accuracy_score(y, (p >= 0.5).astype(int))),
        n=int(len(y)),
    )


def _train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train_df = df[df["date"].dt.year < 2022].copy()
    val_df = df[df["date"].dt.year >= 2022].copy()
    return train_df, val_df


def _train_calib_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-safe split inside train period:
    - train_base: year < 2021
    - calib:      year == 2021
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train_base = df[df["date"].dt.year < 2021].copy()
    calib = df[df["date"].dt.year == 2021].copy()
    return train_base, calib


def _feature_cols(df: pd.DataFrame, include_market: bool) -> List[str]:
    meta_cols = ["date", "surface", "playerA", "playerB", "y"]
    if include_market:
        exclude = meta_cols
    else:
        exclude = meta_cols + ["oddsA", "oddsB", "pA_market", "pB_market", "p_diff", "logit_pA_market", "has_market"]
    return [c for c in df.columns if c not in exclude]


def train_and_select() -> Tuple[Path, Path, Path]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_DATA_PATH}")

    print(f"[best] Reading train dataset: {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    train_df, val_df = _train_val_split(df)
    print(f"[best] Train rows={len(train_df):,} Val rows={len(val_df):,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Candidate 1: Logistic Regression (no market) ---
    feat_nomarket = _feature_cols(df, include_market=False)
    imp1 = SimpleImputer(strategy="median")
    Xtr1 = imp1.fit_transform(train_df[feat_nomarket])
    Xv1 = imp1.transform(val_df[feat_nomarket])
    ytr = train_df["y"].astype(int).values
    yv = val_df["y"].astype(int).values

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, n_jobs=-1, solver="lbfgs"),
    )
    lr.fit(Xtr1, ytr)
    p1 = lr.predict_proba(Xv1)[:, 1]
    m1 = _eval(yv, p1)
    print(f"[best] LR(no-market) logloss={m1.logloss:.6f} brier={m1.brier:.6f} acc={m1.acc:.6f}")

    # --- Candidate 2: HistGradientBoosting (no market) ---
    hgb_grid = [
        # (learning_rate, max_depth, max_leaf_nodes, min_samples_leaf, l2_regularization, max_iter)
        (0.05, 6, 31, 50, 0.0, 400),
        (0.05, 4, 31, 50, 0.0, 400),
        (0.03, 6, 31, 50, 0.0, 600),
        (0.03, 4, 31, 50, 0.0, 600),
        (0.05, 6, 63, 50, 0.0, 400),
        (0.03, 6, 63, 50, 0.0, 600),
        (0.05, 6, 31, 100, 0.0, 400),
        (0.03, 6, 31, 100, 0.0, 600),
        (0.05, 6, 31, 50, 0.1, 400),
        (0.03, 6, 31, 50, 0.1, 600),
    ]

    best_hgb_nm = None
    best_hgb_nm_m: Optional[Metrics] = None
    for lr_, md_, mln_, msl_, l2_, it_ in hgb_grid:
        hgb = HistGradientBoostingClassifier(
            learning_rate=lr_,
            max_depth=md_,
            max_leaf_nodes=mln_,
            min_samples_leaf=msl_,
            l2_regularization=l2_,
            max_iter=it_,
            random_state=42,
        )
        hgb.fit(Xtr1, ytr)
        p2 = hgb.predict_proba(Xv1)[:, 1]
        m2 = _eval(yv, p2)
        if (best_hgb_nm_m is None) or (m2.logloss < best_hgb_nm_m.logloss):
            best_hgb_nm = hgb
            best_hgb_nm_m = m2
    assert best_hgb_nm_m is not None and best_hgb_nm is not None
    print(f"[best] HGB(no-market best) logloss={best_hgb_nm_m.logloss:.6f} brier={best_hgb_nm_m.brier:.6f} acc={best_hgb_nm_m.acc:.6f}")

    # --- Candidate 3: HistGradientBoosting (with market) ---
    # Evaluate only on rows where market exists to compare fairly vs market.
    feat_market = _feature_cols(df, include_market=True)
    if "has_market" not in df.columns:
        # backward-compat: if dataset not rebuilt yet
        df["has_market"] = ((pd.to_numeric(df.get("oddsA"), errors="coerce").notna()) & (pd.to_numeric(df.get("oddsB"), errors="coerce").notna())).astype(int)
        train_df, val_df = _train_val_split(df)

    market_mask_val = val_df.get("has_market", 0).astype(int).values == 1
    market_mask_tr = train_df.get("has_market", 0).astype(int).values == 1

    best_model = lr
    best_imputer = imp1
    best_feats = feat_nomarket
    best_name = "LR(no-market)"
    best_score = m1.logloss

    # prefer better no-market among LR/HGB
    if best_hgb_nm_m.logloss < best_score:
        best_model, best_name, best_score = best_hgb_nm, "HGB(no-market)", best_hgb_nm_m.logloss

    # Now try market model (if we have enough rows)
    if market_mask_tr.sum() >= 5000 and market_mask_val.sum() >= 1000:
        imp3 = SimpleImputer(strategy="median")
        Xtr3 = imp3.fit_transform(train_df.loc[market_mask_tr, feat_market])
        Xv3 = imp3.transform(val_df.loc[market_mask_val, feat_market])
        ytr3 = train_df.loc[market_mask_tr, "y"].astype(int).values
        yv3 = val_df.loc[market_mask_val, "y"].astype(int).values
        best_hgb_m = None
        best_hgb_m_m: Optional[Metrics] = None
        for lr_, md_, mln_, msl_, l2_, it_ in hgb_grid:
            hgb_m = HistGradientBoostingClassifier(
                learning_rate=lr_,
                max_depth=md_,
                max_leaf_nodes=mln_,
                min_samples_leaf=msl_,
                l2_regularization=l2_,
                max_iter=it_,
                random_state=42,
            )
            hgb_m.fit(Xtr3, ytr3)
            p3 = hgb_m.predict_proba(Xv3)[:, 1]
            m3 = _eval(yv3, p3)
            if (best_hgb_m_m is None) or (m3.logloss < best_hgb_m_m.logloss):
                best_hgb_m = hgb_m
                best_hgb_m_m = m3
        assert best_hgb_m is not None and best_hgb_m_m is not None
        print(f"[best] HGB(+market best) on market-rows n={best_hgb_m_m.n:,} logloss={best_hgb_m_m.logloss:.6f} brier={best_hgb_m_m.brier:.6f} acc={best_hgb_m_m.acc:.6f}")

        # Compare vs market baseline on same rows
        pm = pd.to_numeric(val_df.loc[market_mask_val, "pA_market"], errors="coerce").astype(float).values
        mm = _eval(yv3, pm)
        print(f"[best] Market baseline n={mm.n:,} logloss={mm.logloss:.6f} brier={mm.brier:.6f} acc={mm.acc:.6f}")

        # If market-augmented model beats market on logloss, select it (main business objective)
        if best_hgb_m_m.logloss < mm.logloss:
            best_model = best_hgb_m
            best_imputer = imp3
            best_feats = feat_market
            best_name = "HGB(+market)"
            best_score = best_hgb_m_m.logloss

    # --- Calibration (optional but often improves logloss) ---
    # We calibrate only if we have enough 2021 matches in the relevant training slice.
    def _maybe_calibrate(model, imputer, feats: List[str], use_market_mask: bool) -> Tuple[object, float, str]:
        train_base, calib = _train_calib_split(df)
        if len(calib) < 2000 or len(train_base) < 10000:
            return model, float("inf"), "no-calib"

        if use_market_mask and "has_market" in train_base.columns and "has_market" in calib.columns:
            train_base = train_base[train_base["has_market"].astype(int) == 1].copy()
            calib = calib[calib["has_market"].astype(int) == 1].copy()
            if len(calib) < 1000 or len(train_base) < 5000:
                return model, float("inf"), "no-calib"

        Xtr = imputer.fit_transform(train_base[feats])
        ytr2 = train_base["y"].astype(int).values
        Xc = imputer.transform(calib[feats])
        yc = calib["y"].astype(int).values

        # refit base model on train_base
        model.fit(Xtr, ytr2)

        best_cal = None
        best_ll = float("inf")
        best_kind = "no-calib"
        # Avoid deprecated cv="prefit" when possible
        try:
            from sklearn.frozen import FrozenEstimator  # type: ignore
            frozen = FrozenEstimator(model)
            prefit_est = frozen
            cv = None
        except Exception:
            prefit_est = model
            cv = "prefit"

        for method in ["sigmoid", "isotonic"]:
            cal = CalibratedClassifierCV(prefit_est, method=method, cv=cv)  # type: ignore[arg-type]
            cal.fit(Xc, yc)
            # evaluate on val (same imputer already fitted on train_base)
            Xv = imputer.transform(val_df[feats])
            pv = cal.predict_proba(Xv)[:, 1]
            ll = float(log_loss(yv, pv))
            if ll < best_ll:
                best_ll = ll
                best_cal = cal
                best_kind = method
        if best_cal is None:
            return model, float("inf"), "no-calib"
        return best_cal, best_ll, best_kind

    use_market = best_name == "HGB(+market)"
    calibrated_model, cal_ll, cal_kind = _maybe_calibrate(best_model, best_imputer, best_feats, use_market_mask=use_market)
    if cal_ll < best_score:
        best_model = calibrated_model
        best_name = f"{best_name}+cal({cal_kind})"
        best_score = cal_ll

    print(f"[best] Selected: {best_name} (objective logloss={best_score:.6f})")

    model_path = MODELS_DIR / "logreg_final.pkl"   # keep existing filenames for compatibility
    imputer_path = MODELS_DIR / "imputer_final.pkl"
    feature_cols_path = MODELS_DIR / "feature_columns.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(best_imputer, imputer_path)
    feature_cols_path.write_text("\n".join(best_feats) + "\n", encoding="utf-8")

    print(f"[best] Saved model: {model_path}")
    print(f"[best] Saved imputer: {imputer_path}")
    print(f"[best] Saved features: {feature_cols_path} (n={len(best_feats)})")

    return model_path, imputer_path, feature_cols_path


if __name__ == "__main__":
    train_and_select()

