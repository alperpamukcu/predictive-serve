import pandas as pd
import joblib
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

from src.utils.config import PROCESSED_DIR, MODELS_DIR


VAL_START_DATE = "2022-01-01"  # 2022 ve sonrası validation


def load_artifacts():
    """Eğitilmiş modeli, imputeri ve feature kolon listesini yükler."""
    feature_cols_path = MODELS_DIR / "feature_columns.txt"
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    model = joblib.load(MODELS_DIR / "logreg_final.pkl")
    imputer = joblib.load(MODELS_DIR / "imputer_final.pkl")

    return model, imputer, feature_cols


def load_data():
    """train_dataset.csv dosyasını okur ve tarih kolonunu datetime'e çevirir."""
    train_path = PROCESSED_DIR / "train_dataset.csv"
    df = pd.read_csv(train_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_model_predictions(df, model, imputer, feature_cols):
    """Verilen dataframe için model olasılığını hesaplar ve p_model kolonunu ekler."""
    X = df[feature_cols]
    X_imp = imputer.transform(X)
    p_model = model.predict_proba(X_imp)[:, 1]

    df_with_pred = df.copy()
    df_with_pred["p_model"] = p_model
    return df_with_pred


def evaluate_on_val(df_with_pred):
    """Validation set üzerinde model ve market performansını karşılaştırır."""
    val_df = df_with_pred[df_with_pred["date"] >= VAL_START_DATE].copy()

    y_true = val_df["y"].values
    p_model = val_df["p_model"].values
    p_market = val_df["pA_market"].values

    results = {}

    # Model
    results["logloss_model"] = log_loss(y_true, p_model)
    results["brier_model"] = brier_score_loss(y_true, p_model)
    results["acc_model"] = accuracy_score(y_true, p_model >= 0.5)

    # Bahis şirketi
    results["logloss_market"] = log_loss(y_true, p_market)
    results["brier_market"] = brier_score_loss(y_true, p_market)
    results["acc_market"] = accuracy_score(y_true, p_market >= 0.5)

    return val_df, results


def main():
    print("[eval] Veriler yükleniyor...")
    df = load_data()

    print("[eval] Model artefact'leri yükleniyor...")
    model, imputer, feature_cols = load_artifacts()

    print("[eval] Olasılık tahminleri hesaplanıyor (p_model)...")
    df_with_pred = add_model_predictions(df, model, imputer, feature_cols)

    print("[eval] Validation set üzerinde model vs market kıyaslanıyor...")
    val_df, results = evaluate_on_val(df_with_pred)

    # Sonuçları kaydet
    out_path = PROCESSED_DIR / "val_predictions.csv"
    val_df.to_csv(out_path, index=False)
    print(f"[eval] Validation tahminleri kaydedildi: {out_path}")

    # Metrikleri yazdır
    print("\n[eval] Sonuçlar (2022+ validation):")
    print(
        f"  Model  - logloss: {results['logloss_model']:.6f}, "
        f"brier: {results['brier_model']:.6f}, "
        f"accuracy: {results['acc_model']:.6f}"
    )
    print(
        f"  Market - logloss: {results['logloss_market']:.6f}, "
        f"brier: {results['brier_market']:.6f}, "
        f"accuracy: {results['acc_market']:.6f}"
    )


if __name__ == "__main__":
    main()
