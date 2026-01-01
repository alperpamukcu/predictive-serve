# src/models/score_all_matches.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from src.utils.config import PROCESSED_DIR, MODELS_DIR
from src.utils.feature_utils import load_feature_list


TRAIN_PATH = PROCESSED_DIR / "train_dataset.csv"
ALL_PRED_PATH = PROCESSED_DIR / "all_predictions.csv"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.txt"
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"
def main() -> None:
    print(f"[score_all] Train dataset okunuyor: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    feature_cols = load_feature_list(FEATURE_LIST_PATH)
    print(f"[score_all] Feature sayısı: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df["y"].astype(int).values

    print("[score_all] Model ve imputer yükleniyor...")
    model = load(MODEL_PATH)
    imputer = load(IMPUTER_PATH)

    X_imp = imputer.transform(X)
    p_model = model.predict_proba(X_imp)[:, 1]

    # Market olasılığı varsa edge hesapla
    if "pA_market" in df.columns:
        p_market = df["pA_market"].astype(float).values
        edge = p_model - p_market
    else:
        p_market = np.full_like(p_model, np.nan, dtype=float)
        edge = np.full_like(p_model, np.nan, dtype=float)

    out = df[
        [
            "date",
            "surface",
            "playerA",
            "playerB",
            "y",
        ]
    ].copy()
    out["p_model"] = p_model
    out["pA_market"] = p_market
    out["edge"] = edge

    ALL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ALL_PRED_PATH, index=False)

    print(
        f"[score_all] Tüm maçlara ait tahminler kaydedildi: "
        f"{ALL_PRED_PATH} (rows={len(out)})"
    )


if __name__ == "__main__":
    main()
