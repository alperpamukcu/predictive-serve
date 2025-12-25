# src/features/sets.py

import pandas as pd
from collections import defaultdict

from src.utils.config import PROCESSED_DIR, RAW_DIR


def make_player_stats():
    """Her oyuncu için tutulacak kümülatif set istatistikleri."""
    return {
        "sets_won": 0,
        "sets_lost": 0,
        "sets_won_bo3": 0,
        "sets_lost_bo3": 0,
        "sets_won_bo5": 0,
        "sets_lost_bo5": 0,
    }


def load_bo5_tournaments_from_raw():
    """
    allyears.csv içinden 'Best of' = 5 olan tüm turnuva isimlerini okuyup
    lowercase bir set olarak döndürür.
    """
    allyears_path = RAW_DIR / "allyears.csv"
    if not allyears_path.exists():
        print("[sets] allyears.csv bulunamadı, raw'dan bo5 turnuva listesi yüklenemedi.")
        return set()

    try:
        raw = pd.read_csv(
            allyears_path,
            usecols=["Tournament", "Best of"],
            low_memory=False,
        )
    except Exception as e:
        print(f"[sets] allyears.csv okunurken hata: {e}")
        return set()

    raw = raw.dropna(subset=["Tournament", "Best of"])
    try:
        raw["Best of"] = raw["Best of"].astype(int)
    except Exception:
        raw = raw[raw["Best of"] == 5]

    bo5_df = raw[raw["Best of"] == 5].copy()

    bo5_names = (
        bo5_df["Tournament"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    bo5_set = set(bo5_names)
    print(f"[sets] Raw veriden {len(bo5_set)} adet best-of-5 turnuva bulundu.")
    return bo5_set


def infer_best_of(row, set_w_cols, has_best_of, has_series, has_tournament, bo5_tournaments):
    """
    Maçın 3 setlik mi 5 setlik mi olduğunu tahmin et.
    Önce doğrudan 'Best of' kolonu, sonra turnuva adı, sonra set sayısı / seri kullanılır.
    """
    total_sets = row[set_w_cols].notna().sum()

    # 1) Eğer 'Best of' kolonu varsa, onu kullan
    if has_best_of:
        try:
            bo = int(row["Best of"])
            if bo in (3, 5):
                return bo
        except Exception:
            pass

    # 2) Turnuva adına göre (allyears'tan çıkardığımız BO5 listesi)
    if has_tournament:
        tourn = row["Tournament"]
        if isinstance(tourn, str):
            t_name = tourn.strip().lower()
            if t_name in bo5_tournaments:
                return 5

    # 3) Set sayısından kesin çıkarabildiklerimiz
    if total_sets >= 4:
        return 5  # 4 veya 5 set oynanmışsa bu maç best-of-5'tir
    if total_sets <= 2:
        return 3  # 2 setlik maç pratikte best-of-3 kabul edilebilir

    # 4) 3 setlik maçlar için Series bilgisi varsa
    if has_series:
        series = row["Series"]
        if isinstance(series, str) and "Grand Slam" in series:
            return 5

    # 5) Varsayılan: best-of-3
    return 3


def add_set_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    W1-L1... set skorlarından, oyuncuların geçmiş maçlara göre
    set kazanma oranlarını (genel, BO3, BO5) çıkarır.

    ÖNEMLİ: Her maç için feature'lar o maça KADAR olan istatistiklerden hesaplanır.
    """
    # Mevcut W/L set kolonlarını tespit et
    set_w_cols = [c for c in [f"W{i}" for i in range(1, 6)] if c in df.columns]
    set_l_cols = [c for c in [f"L{i}" for i in range(1, 6)] if c in df.columns]

    if not set_w_cols or not set_l_cols:
        print("[sets] W/L set kolonları bulunamadı, set-based feature'lar eklenmedi.")
        return df

    has_best_of = "Best of" in df.columns
    has_series = "Series" in df.columns
    has_tournament = "Tournament" in df.columns

    # Raw allyears'tan bo5 turnuva listesi
    bo5_tournaments = load_bo5_tournaments_from_raw()

    # Tarihe göre sırala
    df = df.sort_values("date").reset_index(drop=True)

    stats = defaultdict(make_player_stats)

    set_winrate_overallA = []
    set_winrate_overallB = []
    set_winrate_overall_diff = []

    set_winrate_bo3A = []
    set_winrate_bo3B = []
    set_winrate_bo3_diff = []

    set_winrate_bo5A = []
    set_winrate_bo5B = []
    set_winrate_bo5_diff = []

    inferred_best_of_list = []

    def safe_rate(won, lost):
        total = won + lost
        if total > 0:
            return won / total
        return float("nan")

    for _, row in df.iterrows():
        pA = row["playerA"]
        pB = row["playerB"]
        y = row["y"]  # 1: A kazandı, 0: B kazandı

        bo = infer_best_of(row, set_w_cols, has_best_of, has_series, has_tournament, bo5_tournaments)
        inferred_best_of_list.append(bo)

        sA = stats[pA]
        sB = stats[pB]

        # --- ÖNCE: Bu maça kadar olan istatistiklerden feature'ları hesapla ---

        # Genel set winrate
        rate_overall_A = safe_rate(sA["sets_won"], sA["sets_lost"])
        rate_overall_B = safe_rate(sB["sets_won"], sB["sets_lost"])

        set_winrate_overallA.append(rate_overall_A)
        set_winrate_overallB.append(rate_overall_B)
        set_winrate_overall_diff.append(rate_overall_A - rate_overall_B)

        # BO3 set winrate
        rate_bo3_A = safe_rate(sA["sets_won_bo3"], sA["sets_lost_bo3"])
        rate_bo3_B = safe_rate(sB["sets_won_bo3"], sB["sets_lost_bo3"])

        set_winrate_bo3A.append(rate_bo3_A)
        set_winrate_bo3B.append(rate_bo3_B)
        set_winrate_bo3_diff.append(rate_bo3_A - rate_bo3_B)

        # BO5 set winrate
        rate_bo5_A = safe_rate(sA["sets_won_bo5"], sA["sets_lost_bo5"])
        rate_bo5_B = safe_rate(sB["sets_won_bo5"], sB["sets_lost_bo5"])

        set_winrate_bo5A.append(rate_bo5_A)
        set_winrate_bo5B.append(rate_bo5_B)
        set_winrate_bo5_diff.append(rate_bo5_A - rate_bo5_B)

        # --- SONRA: Bu maçın sonucuna göre istatistikleri güncelle ---

        total_sets = row[set_w_cols].notna().sum()

        # Bu maçta kazanan/kaybeden kim?
        if y == 1:
            winner, loser = pA, pB
        else:
            winner, loser = pB, pA

        # Bu maçta kazananın kazandığı set sayısı (bo3 -> 2, bo5 -> 3, fail-safe)
        if bo == 3:
            winner_sets = min(2, max(1, total_sets))
        else:  # bo == 5
            winner_sets = min(3, max(1, total_sets))

        loser_sets = max(0, total_sets - winner_sets)

        # Genel set istatistikleri
        stats[winner]["sets_won"] += winner_sets
        stats[winner]["sets_lost"] += loser_sets

        stats[loser]["sets_won"] += loser_sets
        stats[loser]["sets_lost"] += winner_sets

        # Best-of'e göre BO3 / BO5 istatistikleri
        if bo == 3:
            stats[winner]["sets_won_bo3"] += winner_sets
            stats[winner]["sets_lost_bo3"] += loser_sets
            stats[loser]["sets_won_bo3"] += loser_sets
            stats[loser]["sets_lost_bo3"] += winner_sets
        elif bo == 5:
            stats[winner]["sets_won_bo5"] += winner_sets
            stats[winner]["sets_lost_bo5"] += loser_sets
            stats[loser]["sets_won_bo5"] += loser_sets
            stats[loser]["sets_lost_bo5"] += winner_sets

    # Hesaplanan feature'ları dataframe'e ekleyelim
    df["inferred_best_of"] = inferred_best_of_list

    df["set_winrate_overallA"] = set_winrate_overallA
    df["set_winrate_overallB"] = set_winrate_overallB
    df["set_winrate_overall_diff"] = set_winrate_overall_diff

    df["set_winrate_bo3A"] = set_winrate_bo3A
    df["set_winrate_bo3B"] = set_winrate_bo3B
    df["set_winrate_bo3_diff"] = set_winrate_bo3_diff

    df["set_winrate_bo5A"] = set_winrate_bo5A
    df["set_winrate_bo5B"] = set_winrate_bo5B
    df["set_winrate_bo5_diff"] = set_winrate_bo5_diff

    return df


def main():
    input_path = PROCESSED_DIR / "matches_with_elo_form.csv"
    output_path = PROCESSED_DIR / "matches_with_elo_form_sets.csv"

    print(f"[sets] Reading matches_with_elo_form from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["date"])

    df_with_sets = add_set_based_features(df)

    df_with_sets.to_csv(output_path, index=False)
    print(
        f"[sets] Saved matches with Elo + form + set features to: "
        f"{output_path} (rows={len(df_with_sets)})"
    )


if __name__ == "__main__":
    main()
