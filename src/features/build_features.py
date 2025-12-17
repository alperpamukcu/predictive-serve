# src/features/build_features.py

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR

INPUT_PATH = PROCESSED_DIR / "matches_with_elo_form.csv"
OUTPUT_PATH = PROCESSED_DIR / "train_dataset.csv"


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Verilen aday kolon isimlerinden hangisi dataframe'de varsa onu döner."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# 1) H2H FEATURE'LARI
def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Head-to-Head (H2H) feature'larını hesaplar.

    Varsayım: matches_with_elo_form.csv'de playerA = maçı kazanan,
              playerB = kaybeden oyuncu.
    """
    df = df.copy()

    if "date" not in df.columns or "playerA" not in df.columns or "playerB" not in df.columns:
        print("[h2h] Gerekli kolonlar bulunamadı, H2H feature'ları eklenmeyecek.")
        return df

    # tarihe göre sırala
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    h2h_matches_before = np.zeros(n, dtype=np.int32)
    h2h_winsA_before = np.zeros(n, dtype=np.int32)
    h2h_winsB_before = np.zeros(n, dtype=np.int32)

    # key: (player_small, player_big) -> (total_matches, wins_for_player_small)
    h2h_stats: Dict[Tuple[str, str], Tuple[int, int]] = {}

    playersA = df["playerA"].astype(str).values
    playersB = df["playerB"].astype(str).values

    for i in range(n):
        a = playersA[i]
        b = playersB[i]

        # oyuncu isimlerini alfabetik sıralayarak unordered bir anahtar oluştur
        if a <= b:
            key = (a, b)
            a_is_small = True
        else:
            key = (b, a)
            a_is_small = False

        total, wins_small = h2h_stats.get(key, (0, 0))

        # Bu maça GELENE KADAR olan istatistikler:
        h2h_matches_before[i] = total

        if total > 0:
            if a_is_small:
                winsA = wins_small
            else:
                winsA = total - wins_small  # küçük oyuncu B ise, A = büyük oyuncu
            winsB = total - winsA
        else:
            winsA = 0
            winsB = 0

        h2h_winsA_before[i] = winsA
        h2h_winsB_before[i] = winsB

        # Şimdi bu maçı istatistiklere EKLE (A kazandı, B kaybetti)
        total_new = total + 1
        if a_is_small:
            wins_small_new = wins_small + 1  # küçük oyuncu A ve kazandı
        else:
            wins_small_new = wins_small      # küçük oyuncu B, A kazandı -> küçük oyuncu kazanmadı

        h2h_stats[key] = (total_new, wins_small_new)

    # winrate hesapla (0'a bölme uyarısını engelleyerek)
    with np.errstate(divide="ignore", invalid="ignore"):
        h2h_winrateA = np.divide(
            h2h_winsA_before,
            h2h_matches_before,
            out=np.zeros_like(h2h_winsA_before, dtype=float),
            where=h2h_matches_before > 0,
        )
        h2h_winrateB = np.divide(
            h2h_winsB_before,
            h2h_matches_before,
            out=np.zeros_like(h2h_winsB_before, dtype=float),
            where=h2h_matches_before > 0,
        )

    df["h2h_matches"] = h2h_matches_before
    df["h2h_winrateA"] = np.where(h2h_matches_before > 0, h2h_winrateA, np.nan)
    df["h2h_winrateB"] = np.where(h2h_matches_before > 0, h2h_winrateB, np.nan)

    return df


# 2) ROUND FEATURE'LARI
def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turnuvanın sadece round bilgisini feature'a çevirir.

    - round_importance: 1-7 arası önem skoru
    - is_final, is_semi, is_quarter: boolean
    """
    df = df.copy()

    round_col = _find_column(df, ["round", "Round", "round_name"])
    if round_col is not None:
        round_raw = df[round_col].astype(str).str.strip().str.upper()

        def map_round_importance(x: str) -> int:
            if x == "F":
                return 7
            if x == "SF":
                return 6
            if x == "QF":
                return 5
            if "R16" in x or x == "R16":
                return 4
            if "R32" in x:
                return 3
            if "R64" in x or "R56" in x or "R48" in x or "R128" in x:
                return 2
            if "RR" in x:  # Round Robin
                return 4
            # default: erken tur
            return 1

        df["round_importance"] = round_raw.map(map_round_importance)
        df["is_final"] = (round_raw == "F").astype(int)
        df["is_semi"] = (round_raw == "SF").astype(int)
        df["is_quarter"] = (round_raw == "QF").astype(int)
    else:
        print("[tournament] Round kolonu bulunamadı, round feature'ları eklenmedi.")

    return df


# 3) A/B PERSPEKTİF ÇEVİRME
def random_flip_perspective(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Başlangıçta playerA = kazanan.
    Satırların yaklaşık yarısında A/B swap edilir:

      - Swap edilmeyenler: y = 1 (A kazandı)
      - Swap edilenler:    y = 0 (A kaybetti, B kazandı)
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(len(df)) < 0.5

    # y etiketi
    df["y"] = 1
    df.loc[flip_mask, "y"] = 0

    # A/B olarak swap edilecek kolon çiftleri
    col_pairs: List[Tuple[str, str]] = [
        ("playerA", "playerB"),
        ("playerA_norm", "playerB_norm"),
        ("eloA", "eloB"),
        ("elo_surfaceA", "elo_surfaceB"),
        ("form_winrateA_5", "form_winrateB_5"),
        ("form_winrateA_10", "form_winrateB_10"),
        ("days_since_lastA", "days_since_lastB"),
        ("matches_last30A", "matches_last30B"),
        ("rankA", "rankB"),
        ("oddsA", "oddsB"),
        # H2H winrate kolonlarını da swap et
        ("h2h_winrateA", "h2h_winrateB"),
    ]

    for colA, colB in col_pairs:
        if colA in df.columns and colB in df.columns:
            tmp = df.loc[flip_mask, colA].copy()
            df.loc[flip_mask, colA] = df.loc[flip_mask, colB]
            df.loc[flip_mask, colB] = tmp

    return df


# 4) MARKET FEATURE'LARI
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bahis oranlarından implied probability feature'ları üretir."""
    df = df.copy()

    invA = 1.0 / df["oddsA"]
    invB = 1.0 / df["oddsB"]
    denom = invA + invB

    df["pA_market"] = invA / denom
    df["pB_market"] = invB / denom

    eps = 1e-6
    df["logit_pA_market"] = np.log(
        (df["pA_market"] + eps) / (1.0 - df["pA_market"] + eps)
    )
    df["p_diff"] = df["pA_market"] - df["pB_market"]

    return df


# 5) DAYS FEATURE'LARI
def clip_days_features(df: pd.DataFrame, max_days: int = 365) -> pd.DataFrame:
    """days_since_last* feature'larını uç değerlerden korumak için kırpar."""
    df = df.copy()

    for col in ["days_since_lastA", "days_since_lastB"]:
        if col in df.columns:
            df[col + "_clipped"] = df[col].clip(lower=0, upper=max_days)

    if "days_since_lastA_clipped" in df.columns and "days_since_lastB_clipped" in df.columns:
        df["days_since_last_diff_clipped"] = (
            df["days_since_lastA_clipped"] - df["days_since_lastB_clipped"]
        )

    return df


# 6) FARK FEATURE'LARI
def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """A ve B arasındaki fark feature'larını ekler."""
    df = df.copy()

    df["elo_diff"] = df["eloA"] - df["eloB"]
    df["elo_surface_diff"] = df["elo_surfaceA"] - df["elo_surfaceB"]

    df["form_winrate_diff_5"] = df["form_winrateA_5"] - df["form_winrateB_5"]
    df["form_winrate_diff_10"] = df["form_winrateA_10"] - df["form_winrateB_10"]

    df["matches_last30_diff"] = df["matches_last30A"] - df["matches_last30B"]

    if "rankA" in df.columns and "rankB" in df.columns:
        df["rank_diff"] = df["rankA"] - df["rankB"]

    # H2H winrate farkı
    if "h2h_winrateA" in df.columns and "h2h_winrateB" in df.columns:
        df["h2h_winrate_diff"] = df["h2h_winrateA"] - df["h2h_winrateB"]

    return df


# 7) ANA PIPELINE
def build_feature_dataset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Tüm feature'ları üretip train_dataset.csv olarak kaydeder."""
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    print(f"[features] Reading matches_with_elo_form from: {input_path}")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 0) Round feature'ları
    df = add_tournament_features(df)

    # 1) H2H feature'ları (playerA = winner, playerB = loser aşamasında)
    df = add_h2h_features(df)

    # 2) Surface one-hot feature'lar
    surface_dummies = pd.get_dummies(df["surface"], prefix="surface")
    df = pd.concat([df, surface_dummies], axis=1)

    # 3) A/B perspektifini rastgele çevir, y etiketini oluştur
    df = random_flip_perspective(df, seed=42)

    # 4) Market odds feature'ları
    df = add_market_features(df)

    # 5) Gün farkı feature'larını kırp ve farklarını ekle
    df = clip_days_features(df, max_days=365)

    # 6) Elo, form, rank, maç sayısı, H2H farkları
    df = add_diff_features(df)

    surface_feature_cols = surface_dummies.columns.tolist()

    # 7) Kullanacağımız ana feature listesi
    feature_cols = [
        # Elo
        "eloA", "eloB", "elo_diff",
        "elo_surfaceA", "elo_surfaceB", "elo_surface_diff",
        # Form
        "form_winrateA_5", "form_winrateB_5", "form_winrate_diff_5",
        "form_winrateA_10", "form_winrateB_10", "form_winrate_diff_10",
        # Dinlenme / yoğunluk
        "days_since_lastA_clipped", "days_since_lastB_clipped", "days_since_last_diff_clipped",
        "matches_last30A", "matches_last30B", "matches_last30_diff",
        # Rank
        "rankA", "rankB", "rank_diff",
        # H2H
        "h2h_matches", "h2h_winrateA", "h2h_winrateB", "h2h_winrate_diff",
        # Round
        "round_importance", "is_final", "is_semi", "is_quarter",
        # Market odds
        "oddsA", "oddsB",
        "pA_market", "pB_market", "p_diff", "logit_pA_market",
    ] + surface_feature_cols  # surface_* feature'ları da ekle

    # Sadece gerçekten var olan kolonları al
    feature_cols = [c for c in feature_cols if c in df.columns]

    cols_to_keep = (
        ["date", "surface", "playerA", "playerB"]
        + feature_cols
        + ["y"]
    )

    df_out = df[cols_to_keep].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(
        f"[features] Saved training dataset to: {output_path} "
        f"(rows={len(df_out)}, features={len(feature_cols)})"
    )
    return output_path


if __name__ == "__main__":
    build_feature_dataset()
