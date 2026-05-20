# src/features/elo.py

import math
import re
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Optional, Tuple, Dict

import pandas as pd

from src.utils.config import PROCESSED_DIR


MATCHES_CLEAN_PATH = PROCESSED_DIR / "matches_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_elo.csv"

# Başlangıç rating'i
BASE_ELO = 1500.0
# Legacy fixed K-factors (kept as the floor of the dynamic schedule).
K_OVERALL = 32.0
K_SURFACE = 24.0


def expected_score(r_a: float, r_b: float) -> float:
    """İki rating arasından A'nın beklenen skorunu hesaplar."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


# ---------------------------------------------------------------------------
# Phase 2 model improvement: dynamic K-factor + margin-of-victory weighting
# (the "538-style" tennis Elo). New players move fast, veterans move slowly;
# a dominant win shifts ratings more than a deciding-set squeaker.
# ---------------------------------------------------------------------------

def dynamic_k(n_matches: int, floor: float) -> float:
    """K-factor that decays with experience: a debutant's rating reacts
    fast, a 400-match veteran barely twitches.
        n=0   -> ~floor + 27
        n=30  -> ~floor + 10
        n=100 -> ~floor + 6
    """
    return floor + 60.0 / ((n_matches + 5) ** 0.5)


def _parse_games(score: object) -> Tuple[Optional[int], Optional[int]]:
    """Total games won by the winner / loser from a '6-4 7-5' style string.
    matches_clean stores the winner's games first."""
    if not isinstance(score, str) or not score.strip():
        return None, None
    wg = lg = 0
    for chunk in score.split():
        m = re.match(r"(\d+)\D+(\d+)", chunk.strip())
        if not m:
            continue
        wg += int(m.group(1))
        lg += int(m.group(2))
    if wg + lg == 0:
        return None, None
    return wg, lg


# --- Tier-specific Elo (Grand Slams + Masters 1000) ----------------------
_BIG_KEYWORDS = (
    "australian open",
    "french open",
    "roland garros",
    "wimbledon",
    "us open",
    "indian wells",
    "miami",
    "monte carlo",
    "monte-carlo",
    "madrid",
    "rome",
    "internazionali",
    "canada",
    "canadian open",
    "national bank",
    "rogers",
    "cincinnati",
    "shanghai",
    "paris masters",
    "bercy",
    "atp finals",
    "tour finals",
    "nitto",
)


def is_big_tournament(name: object) -> bool:
    """True for Grand Slams + ATP Masters 1000 + ATP Finals — the matches
    that decide season titles and produce the strongest skill signal."""
    if not isinstance(name, str):
        return False
    s = name.lower()
    return any(k in s for k in _BIG_KEYWORDS)


def mov_multiplier(score: object) -> float:
    """Margin-of-victory multiplier in roughly [0.80, 1.35].
    A 12-3 straight-sets win weighs ~1.35x; a 20-18 marathon ~0.90x."""
    wg, lg = _parse_games(score)
    if wg is None or lg is None:
        return 1.0
    total = wg + lg
    if total == 0:
        return 1.0
    dominance = wg / total                  # 0.5 (squeaker) .. ~0.85 (blowout)
    mult = 0.85 + (dominance - 0.5) * 2.0    # 0.5 -> 0.85, 0.75 -> 1.35
    return min(1.35, max(0.80, mult))


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
    # Per-player recent Elo history for momentum (Elo now vs ~5 matches ago)
    elo_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=6))
    # Match counts drive the dynamic K-factor schedule.
    match_count: Dict[str, int] = defaultdict(int)
    match_count_surf: Dict[Tuple[str, str], int] = defaultdict(int)
    # Tier-specific Elo: only updates on Grand Slams + Masters 1000.
    big_ratings: Dict[str, float] = defaultdict(lambda: base_elo)
    match_count_big: Dict[str, int] = defaultdict(int)

    eloA_list = []
    eloB_list = []
    elo_surfaceA_list = []
    elo_surfaceB_list = []
    elo_momentumA_list = []
    elo_momentumB_list = []
    elo_bigA_list = []
    elo_bigB_list = []

    def _momentum(player: str, current: float) -> float:
        """Current Elo minus the Elo recorded ~5 matches ago. 0 if too new."""
        hist = elo_history[player]
        if not hist:
            return 0.0
        return current - hist[0]

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
        elo_momentumA_list.append(_momentum(playerA, rA))
        elo_momentumB_list.append(_momentum(playerB, rB))
        elo_bigA_list.append(big_ratings[playerA])
        elo_bigB_list.append(big_ratings[playerB])

        # playerA her zaman kazanan. Margin-of-victory ağırlığı + dinamik K.
        mov = mov_multiplier(row.get("score"))

        # --- Genel Elo ---
        expA = expected_score(rA, rB)
        kA = dynamic_k(match_count[playerA], floor=k_overall - 8.0)
        kB = dynamic_k(match_count[playerB], floor=k_overall - 8.0)
        delta = mov * (1.0 - expA)  # A won: (scoreA - expA) = 1 - expA
        overall_ratings[playerA] = rA + kA * delta
        overall_ratings[playerB] = rB - kB * delta

        # --- Surface Elo ---
        expA_surf = expected_score(rA_surf, rB_surf)
        kA_s = dynamic_k(match_count_surf[keyA_surf], floor=k_surface - 6.0)
        kB_s = dynamic_k(match_count_surf[keyB_surf], floor=k_surface - 6.0)
        delta_s = mov * (1.0 - expA_surf)
        surface_ratings[keyA_surf] = rA_surf + kA_s * delta_s
        surface_ratings[keyB_surf] = rB_surf - kB_s * delta_s

        # --- Tier-specific Elo (only on Grand Slams + Masters 1000) ---
        if is_big_tournament(row.get("tourney")):
            rA_big = big_ratings[playerA]
            rB_big = big_ratings[playerB]
            expA_big = expected_score(rA_big, rB_big)
            kA_b = dynamic_k(match_count_big[playerA], floor=k_surface - 6.0)
            kB_b = dynamic_k(match_count_big[playerB], floor=k_surface - 6.0)
            delta_b = mov * (1.0 - expA_big)
            big_ratings[playerA] = rA_big + kA_b * delta_b
            big_ratings[playerB] = rB_big - kB_b * delta_b
            match_count_big[playerA] += 1
            match_count_big[playerB] += 1

        # Bookkeeping
        match_count[playerA] += 1
        match_count[playerB] += 1
        match_count_surf[keyA_surf] += 1
        match_count_surf[keyB_surf] += 1

        # Record post-match Elo for momentum tracking
        elo_history[playerA].append(overall_ratings[playerA])
        elo_history[playerB].append(overall_ratings[playerB])

    # Yeni kolonları DataFrame'e ekle
    df["eloA"] = eloA_list
    df["eloB"] = eloB_list
    df["elo_surfaceA"] = elo_surfaceA_list
    df["elo_surfaceB"] = elo_surfaceB_list
    df["elo_momentumA"] = elo_momentumA_list
    df["elo_momentumB"] = elo_momentumB_list
    df["elo_bigA"] = elo_bigA_list
    df["elo_bigB"] = elo_bigB_list

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
