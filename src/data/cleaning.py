# src/data/cleaning.py

from pathlib import Path
import pandas as pd

from src.utils.config import PROCESSED_DIR
from src.data.schema import MATCH_COLUMNS

RAW_MATCHES_PATH = PROCESSED_DIR / "matches_allyears.csv"
CLEAN_MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"


def load_raw_matches(path: Path = RAW_MATCHES_PATH) -> pd.DataFrame:
    """Preprocess çıktısını (matches_allyears.csv) okur ve tarihi parse eder."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA'dan çıkan kurallara göre temizlenmiş maç datası üretir.
    """
    df = df.copy()

    # 1) Tarih + oyuncu isimleri olmayan satırları at
    df = df.dropna(subset=["date", "playerA", "playerB"])

    # 2) Yıl filtresi (2000–2025)
    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]

    # 3) Odds'u sayıya çevir
    for col in ["oddsA", "oddsB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Odds'u eksik olan satırları at
    df = df.dropna(subset=["oddsA", "oddsB"])

    # 5) Odds için mantıklı aralık filtresi
    df = df[
        (df["oddsA"] >= 1.01) & (df["oddsA"] <= 100.0) &
        (df["oddsB"] >= 1.01) & (df["oddsB"] <= 100.0)
    ]

    # 6) Tamamlanmamış maçları çıkar (retired, walkover, abandon vb.)
    df["comment"] = df["comment"].fillna("").astype(str)
    bad_mask = (
        df["comment"].str.contains("ret", case=False)
        | df["comment"].str.contains("walk", case=False)
        | df["comment"].str.contains("abandon", case=False)
    )
    df = df[~bad_mask]

    # Geçici year kolonunu kaldır
    df = df.drop(columns=["year"])

    # Kolon sırasını standart hale getir
    df = df[MATCH_COLUMNS]

    return df


def build_clean_matches(
    input_path: Path = RAW_MATCHES_PATH,
    output_path: Path = CLEAN_MATCHES_PATH,
) -> Path:
    """Diskten ham maçları okur, temizler ve matches_clean.csv olarak yazar."""
    df_raw = load_raw_matches(input_path)
    df_clean = clean_matches(df_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[cleaning] Saved clean matches to: {output_path} (rows={len(df_clean)})")
    return output_path


if __name__ == "__main__":
    build_clean_matches()
