# src/features/form.py

from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import pandas as pd
import numpy as np

from src.utils.config import PROCESSED_DIR

INPUT_PATH = PROCESSED_DIR / "matches_with_elo.csv"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_elo_form.csv"


def compute_form_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 10,
    recent_days: int = 30,
) -> pd.DataFrame:
    """
    Her maç için oyuncuların form ve yorgunluk feature'larını hesaplar.

    Üretilen kolonlar:
      - form_winrateA_5,  form_winrateB_5
      - form_winrateA_10, form_winrateB_10
      - days_since_lastA, days_since_lastB
      - matches_last30A,  matches_last30B
    """
    df = df.copy()

    # Tarihe göre sırala (eskiden yeniye)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Elo hesaplamasında oluşturduğumuz normalize isim kolonlarını kullan
    if "playerA_norm" not in df.columns:
        df["playerA_norm"] = df["playerA"].astype(str).str.strip().str.lower()
    if "playerB_norm" not in df.columns:
        df["playerB_norm"] = df["playerB"].astype(str).str.strip().str.lower()

    # Her oyuncu için:
    # - Son sonuçlar (1 = win, 0 = loss) -> max window_long kadar tutacağız
    # - Tüm geçmiş maç tarihleri -> matches_last30 için
    # - Son maç tarihi -> days_since_last için
    results_history: Dict[str, Deque[int]] = defaultdict(
        lambda: deque(maxlen=window_long)
    )
    match_dates: Dict[str, List[pd.Timestamp]] = defaultdict(list)
    last_date: Dict[str, pd.Timestamp] = defaultdict(lambda: pd.NaT)

    # Çıkacak kolonları listede toplayacağız
    form_winrateA_5: List[float] = []
    form_winrateB_5: List[float] = []
    form_winrateA_10: List[float] = []
    form_winrateB_10: List[float] = []
    days_since_lastA: List[float] = []
    days_since_lastB: List[float] = []
    matches_last30A: List[int] = []
    matches_last30B: List[int] = []

    for _, row in df.iterrows():
        date = row["date"]
        playerA = row["playerA_norm"]
        playerB = row["playerB_norm"]

        # --- A oyuncusu için ---
        histA = results_history[playerA]
        datesA = match_dates[playerA]
        lastA = last_date[playerA]

        # Son 5 / 10 maç winrate
        def winrate_from_hist(hist: Deque[int], k: int) -> float:
            if not hist:
                return np.nan
            vals = list(hist)[-k:]
            return sum(vals) / len(vals) if vals else np.nan

        wrA5 = winrate_from_hist(histA, window_short)
        wrA10 = winrate_from_hist(histA, window_long)

        # Son maçtan bu maça gün farkı
        if pd.isna(lastA):
            dA = np.nan
        else:
            dA = (date - lastA).days

        # Son 30 günde kaç maç?
        if datesA:
            cutoffA = date - pd.Timedelta(days=recent_days)
            mA = sum(1 for d in datesA if d >= cutoffA)
        else:
            mA = 0

        # --- B oyuncusu için ---
        histB = results_history[playerB]
        datesB = match_dates[playerB]
        lastB = last_date[playerB]

        wrB5 = winrate_from_hist(histB, window_short)
        wrB10 = winrate_from_hist(histB, window_long)

        if pd.isna(lastB):
            dB = np.nan
        else:
            dB = (date - lastB).days

        if datesB:
            cutoffB = date - pd.Timedelta(days=recent_days)
            mB = sum(1 for d in datesB if d >= cutoffB)
        else:
            mB = 0

        # Bu maç için feature'ları kaydet (güncellemeden ÖNCE)
        form_winrateA_5.append(wrA5)
        form_winrateB_5.append(wrB5)
        form_winrateA_10.append(wrA10)
        form_winrateB_10.append(wrB10)
        days_since_lastA.append(dA)
        days_since_lastB.append(dB)
        matches_last30A.append(mA)
        matches_last30B.append(mB)

        # Şimdi sonucu tarihle birlikte geçmişe ekleyelim
        # Bu maçta playerA kazandı, playerB kaybetti
        results_history[playerA].append(1)
        results_history[playerB].append(0)

        match_dates[playerA].append(date)
        match_dates[playerB].append(date)

        last_date[playerA] = date
        last_date[playerB] = date

    # Kolonları DataFrame'e ekleyelim
    df["form_winrateA_5"] = form_winrateA_5
    df["form_winrateB_5"] = form_winrateB_5
    df["form_winrateA_10"] = form_winrateA_10
    df["form_winrateB_10"] = form_winrateB_10
    df["days_since_lastA"] = days_since_lastA
    df["days_since_lastB"] = days_since_lastB
    df["matches_last30A"] = matches_last30A
    df["matches_last30B"] = matches_last30B

    return df


def build_matches_with_form(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """matches_with_elo.csv'ye form feature'larını ekleyip yeni dosya oluşturur."""
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    print(f"[form] Reading matches_with_elo from: {input_path}")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df_out = compute_form_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[form] Saved matches with Elo + form to: {output_path} (rows={len(df_out)})")

    return output_path


if __name__ == "__main__":
    build_matches_with_form()
