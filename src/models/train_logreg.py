# src/models/train_logreg.py
"""
Baseline logistic-regression trainer with a strict no-market feature set
and a proper train/val/test split.
    train  : year < 2022
    val    : 2022 <= year < 2025
    test   : year >= 2025  (held-out — never tuned against)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import MODELS_DIR, PROCESSED_DIR
from src.utils.feature_utils import select_model_features

TRAIN_DATA_PATH = PROCESSED_DIR / "train_dataset.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"


def _eval(y, p) -> dict:
    return {
        "n": int(len(y)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def train_logistic_regression() -> Tuple[Path, Path, Path]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Train dataset bulunamadı: {TRAIN_DATA_PATH}")

    print(f"[logreg] Reading train dataset from: {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    feature_cols = select_model_features(list(df.columns), include_market=False)
    print(f"[logreg] Feature count (no market): {len(feature_cols)}")

    yr = df["date"].dt.year
    train_df = df[yr < 2022].copy()
    val_df = df[(yr >= 2022) & (yr < 2025)].copy()
    test_df = df[yr >= 2025].copy()
    print(f"[logreg] Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[feature_cols])
    X_val = imputer.transform(val_df[feature_cols])
    X_test = imputer.transform(test_df[feature_cols]) if len(test_df) else None

    y_train = train_df["y"].astype(int).values
    y_val = val_df["y"].astype(int).values
    y_test = test_df["y"].astype(int).values if len(test_df) else None

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, n_jobs=-1, solver="lbfgs"),
    )
    print("[logreg] Training...")
    model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]
    val_metrics = _eval(y_val, p_val)
    print(f"[logreg] Val   {val_metrics}")

    test_metrics = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        p_test = model.predict_proba(X_test)[:, 1]
        test_metrics = _eval(y_test, p_test)
        print(f"[logreg] Test  {test_metrics}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "logreg_final.pkl"
    imputer_path = MODELS_DIR / "imputer_final.pkl"
    feature_cols_path = MODELS_DIR / "feature_columns.txt"

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)
    feature_cols_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    METRICS_PATH.write_text(
        json.dumps(
            {"model": "logreg", "validation": val_metrics, "test": test_metrics},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[logreg] Saved model -> {model_path}")
    print(f"[logreg] Saved imputer -> {imputer_path}")
    print(f"[logreg] Saved features -> {feature_cols_path}")
    print(f"[logreg] Metrics -> {METRICS_PATH}")

    return model_path, imputer_path, feature_cols_path


if __name__ == "__main__":
    train_logistic_regression()
