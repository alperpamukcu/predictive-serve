# src/features/elo.py

from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict

import pandas as pd

from src.utils.config import PROCESSED_DIR


MATCHES_CLEAN_PATH = PROCESSED_DIR / "matches_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_elo.csv"

# Başlangıç rating'i
BASE_ELO = 1500.0
# Güncelleme katsayısı (K-faktörü)
K_OVERALL = 32.0
K_SURFACE = 24.0


def expected_score(r_a: float, r_b: float) -> float:
    """İki rating arasından A'nın beklenen skorunu hesaplar."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def compute_elo_for_matches(
    df: pd.DataFrame,
    base_elo: float = BASE_ELO,
    k_overall: float = K_OVERALL,
    k_surface: float = K_SURFACE,
) -> pd.DataFrame:
    """
    matches_clean DataFrame'i üzerinde Elo ve surface Elo hesaplar.

    Varsayımlar:
      - playerA her zaman maçı kazanan oyuncu (winner = 'A')
      - playerA_norm ve playerB_norm kolonları mevcut
      - surface kolonunda Hard/Clay/Grass/Carpet/Indoor gibi değerler var
    """
    df = df.copy()

    # Tarihe göre sırala (eskiden yeniye)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Oyuncu isimlerini normalize eden kolonlar yoksa oluştur
    if "playerA_norm" not in df.columns:
        df["playerA_norm"] = df["playerA"].astype(str).str.strip().str.lower()
    if "playerB_norm" not in df.columns:
        df["playerB_norm"] = df["playerB"].astype(str).str.strip().str.lower()

    # Rating sözlükleri
    overall_ratings: Dict[str, float] = defaultdict(lambda: base_elo)
    surface_ratings: Dict[Tuple[str, str], float] = defaultdict(lambda: base_elo)

    eloA_list = []
    eloB_list = []
    elo_surfaceA_list = []
    elo_surfaceB_list = []

    for _, row in df.iterrows():
        playerA = row["playerA_norm"]
        playerB = row["playerB_norm"]
        surface = row.get("surface") or "Unknown"

        keyA_surf = (playerA, surface)
        keyB_surf = (playerB, surface)

        # Maçtan ÖNCEKİ rating'ler
        rA = overall_ratings[playerA]
        rB = overall_ratings[playerB]
        rA_surf = surface_ratings[keyA_surf]
        rB_surf = surface_ratings[keyB_surf]

        eloA_list.append(rA)
        eloB_list.append(rB)
        elo_surfaceA_list.append(rA_surf)
        elo_surfaceB_list.append(rB_surf)

        # Maç sonucu: playerA her zaman kazanan (winner = 'A')
        scoreA = 1.0
        scoreB = 0.0

        # Genel Elo güncellemesi
        expA = expected_score(rA, rB)
        expB = 1.0 - expA

        overall_ratings[playerA] = rA + k_overall * (scoreA - expA)
        overall_ratings[playerB] = rB + k_overall * (scoreB - expB)

        # Surface Elo güncellemesi
        expA_surf = expected_score(rA_surf, rB_surf)
        expB_surf = 1.0 - expA_surf

        surface_ratings[keyA_surf] = rA_surf + k_surface * (scoreA - expA_surf)
        surface_ratings[keyB_surf] = rB_surf + k_surface * (scoreB - expB_surf)

    # Yeni kolonları DataFrame'e ekle
    df["eloA"] = eloA_list
    df["eloB"] = eloB_list
    df["elo_surfaceA"] = elo_surfaceA_list
    df["elo_surfaceB"] = elo_surfaceB_list

    return df


def build_matches_with_elo(
    input_path: Path = MATCHES_CLEAN_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """matches_clean.csv'yi okuyup Elo kolonlarını ekler ve yeni dosya olarak yazar."""
    if not input_path.exists():
        raise FileNotFoundError(f"Temiz maç datası bulunamadı: {input_path}")

    print(f"[elo] Reading clean matches from: {input_path}")
    df = pd.read_csv(input_path)

    df_with_elo = compute_elo_for_matches(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_elo.to_csv(output_path, index=False)
    print(f"[elo] Saved matches with Elo to: {output_path} (rows={len(df_with_elo)})")

    return output_path


if __name__ == "__main__":
    build_matches_with_elo()
