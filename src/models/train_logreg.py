# src/models/train_logreg.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

from src.utils.config import PROCESSED_DIR

# PROCESSED_DIR = <PROJECT_ROOT>/data/processed
# Buradan PROJECT_ROOT'a çıkıp models klasörünü tanımlıyoruz:
MODELS_DIR = PROCESSED_DIR.parent.parent / "models"

TRAIN_DATA_PATH = PROCESSED_DIR / "train_dataset.csv"


def train_logistic_regression() -> tuple[Path, Path, Path]:
    """
    Logistic Regression modelini eğitir, validation performansını yazar
    ve modeli + imputer'ı + feature list'ini diske kaydeder.

    Dönüş:
        (model_path, imputer_path, feature_columns_path)
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Train dataset bulunamadı: {TRAIN_DATA_PATH}")

    print(f"[logreg] Reading train dataset from: {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Meta kolonlar (feature olmayanlar)
    meta_cols = ["date", "surface", "playerA", "playerB", "y"]

    # Feature kolonları: meta olmayan tüm kolonlar
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"[logreg] Toplam feature sayısı: {len(feature_cols)}")

    # Zaman bazlı train/validation ayrımı
    train_df = df[df["date"].dt.year < 2022].copy()
    val_df = df[df["date"].dt.year >= 2022].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["y"]

    X_val = val_df[feature_cols]
    y_val = val_df["y"]

    print(f"[logreg] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Eksik değerleri median ile doldur
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    # Logistic Regression modeli
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
    )

    print("[logreg] Model eğitiliyor...")
    model.fit(X_train_imp, y_train)

    # Validation performansı
    val_proba = model.predict_proba(X_val_imp)[:, 1]

    logloss_val = log_loss(y_val, val_proba)
    brier_val = brier_score_loss(y_val, val_proba)
    acc_val = accuracy_score(y_val, (val_proba >= 0.5).astype(int))

    print(f"[logreg] Validation logloss : {logloss_val:.6f}")
    print(f"[logreg] Validation brier   : {brier_val:.6f}")
    print(f"[logreg] Validation accuracy: {acc_val:.6f}")

    # MODELS_DIR altında kayıt
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "logreg_final.pkl"
    imputer_path = MODELS_DIR / "imputer_final.pkl"
    feature_cols_path = MODELS_DIR / "feature_columns.txt"

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)

    with feature_cols_path.open("w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print(f"[logreg] Model kaydedildi: {model_path}")
    print(f"[logreg] Imputer kaydedildi: {imputer_path}")
    print(f"[logreg] Feature listesi kaydedildi: {feature_cols_path}")

    return model_path, imputer_path, feature_cols_path


if __name__ == "__main__":
    train_logistic_regression()
