# src/data/preprocess.py

"""
allyears.csv dosyasını okuyup proje için standart "matches" formatına çeviren script.
Çıktı: data/processed/matches_allyears.csv
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import ALLYEARS_PATH, PROCESSED_DIR
from src.data.schema import MATCH_COLUMNS


OUTPUT_PATH = PROCESSED_DIR / "matches_allyears.csv"


def _build_score(row: pd.Series) -> Optional[str]:
    """
    allyears setindeki W1..W5, L1..L5 kolonlarından
    "6-4 7-5" gibi bir skor string'i üretir.
    """
    parts = []
    for i in range(1, 6):  # W1..W5, L1..L5
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if pd.isna(w) or pd.isna(l):
            continue
        try:
            w_int = int(w)
            l_int = int(l)
        except (TypeError, ValueError):
            continue
        parts.append(f"{w_int}-{l_int}")
    return " ".join(parts) if parts else None


def _fair_probs_from_odds(odds_a, odds_b):
    """
    Bookmaker marjını kaldırarak odds'tan yaklaşık adil olasılık çıkartır.
    1 / oddsA, 1 / oddsB değerlerini normalize eder.
    """
    import math

    if odds_a is None or odds_b is None:
        return None, None

    try:
        odds_a = float(odds_a)
        odds_b = float(odds_b)
    except (TypeError, ValueError):
        return None, None

    if odds_a <= 0 or odds_b <= 0:
        return None, None

    pa_raw = 1.0 / odds_a
    pb_raw = 1.0 / odds_b
    total = pa_raw + pb_raw
    if total == 0 or math.isinf(total):
        return None, None

    return pa_raw / total, pb_raw / total


def build_matches_from_allyears(
    input_path: Path = ALLYEARS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    allyears.csv dosyasını okuyup proje için standart matches formatına çevirir
    ve data/processed/matches_allyears.csv olarak kaydeder.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"allyears.csv bulunamadı: {input_path}")

    print(f"[preprocess] Reading allyears: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # Gerekli kolonlar:
    required_cols = ["Date", "Tournament", "Surface", "Round", "Winner", "Loser"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Gerekli kolon eksik: {col}")

    has_b365w = "B365W" in df.columns
    has_b365l = "B365L" in df.columns
    has_comment = "Comment" in df.columns
    has_wrank = "WRank" in df.columns
    has_lrank = "LRank" in df.columns

    out = pd.DataFrame()

    # Temel bilgiler
    out["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)
    out["tourney"] = df["Tournament"]
    out["surface"] = df["Surface"]
    out["round"] = df["Round"]

    # Oyuncular ve rank
    out["playerA"] = df["Winner"]
    out["playerB"] = df["Loser"]
    out["rankA"] = df["WRank"] if has_wrank else None
    out["rankB"] = df["LRank"] if has_lrank else None

    # Odds
    out["oddsA"] = df["B365W"] if has_b365w else None
    out["oddsB"] = df["B365L"] if has_b365l else None

    # Skor
    out["score"] = df.apply(_build_score, axis=1)

    # Yorum
    out["comment"] = df["Comment"] if has_comment else None

    # Kaynak dosya
    out["source_file"] = input_path.name

    # allyears'in ATP erkek maçı olduğunu varsayıyoruz (şimdilik)
    out["gender"] = "M"

    # PlayerA = Winner kuralı
    out["winner"] = "A"

    # Normalize isimler
    out["playerA_norm"] = out["playerA"].astype(str).str.strip().str.lower()
    out["playerB_norm"] = out["playerB"].astype(str).str.strip().str.lower()

    # Olasılıklar
    pA_list = []
    pB_list = []
    for a, b in zip(out["oddsA"], out["oddsB"]):
        pa, pb = _fair_probs_from_odds(a, b)
        pA_list.append(pa)
        pB_list.append(pb)
    out["pA_implied_fair"] = pA_list
    out["pB_implied_fair"] = pB_list

    # Kolon sırasını sabitle
    out = out[MATCH_COLUMNS]

    # Çıktı klasörünü oluştur ve kaydet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[preprocess] Saved: {output_path} ({len(out)} rows)")
    return output_path


if __name__ == "__main__":
    build_matches_from_allyears()
