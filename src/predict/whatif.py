# src/predict/whatif.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import numpy as np
import pandas as pd
from joblib import load

from src.utils.config import PROCESSED_DIR, MODELS_DIR


HISTORY_PATH = PROCESSED_DIR / "matches_with_elo_form_sets.csv"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.txt"
MODEL_PATH = MODELS_DIR / "logreg_final.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer_final.pkl"


@dataclass
class PlayerSnapshot:
    elo: float
    elo_surface: float
    form5: float
    form10: float
    days_since_last: float
    matches_last30: float
    rank: float


def load_feature_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def map_round_importance(round_code: Optional[str]) -> Tuple[float, int, int, int]:
    """build_features.py ile uyumlu round feature'ları döner."""
    if not round_code:
        return np.nan, 0, 0, 0

    x = round_code.strip().upper()

    if x == "F":
        return 7, 1, 0, 0
    if x == "SF":
        return 6, 0, 1, 0
    if x == "QF":
        return 5, 0, 0, 1
    if "R16" in x or x == "R16":
        return 4, 0, 0, 0
    if "R32" in x:
        return 3, 0, 0, 0
    if any(r in x for r in ["R64", "R56", "R48", "R128"]):
        return 2, 0, 0, 0
    if "RR" in x:
        return 4, 0, 0, 0
    return 1, 0, 0, 0


def get_player_snapshot(
    history: pd.DataFrame,
    player: str,
    date: pd.Timestamp,
    surface: Optional[str] = None,
) -> Optional[PlayerSnapshot]:
    """
    Verilen tarihten ÖNCE oynanan maçlardan oyuncunun son snapshot'ını çıkarır.
    Aşırı hassas olmak yerine son maçtaki değerleri kullanıyoruz.
    """
    h = history[
        ((history["playerA"] == player) | (history["playerB"] == player))
        & (history["date"] < date)
    ].sort_values("date")

    if h.empty:
        return None

    last = h.iloc[-1]

    # Oyuncu A mı B mi?
    if last["playerA"] == player:
        prefix = "A"
    else:
        prefix = "B"

    def safe(col: str) -> float:
        c = f"{col}{prefix}"
        return float(last[c]) if c in last and pd.notna(last[c]) else np.nan

    elo = safe("elo")
    elo_surface = safe("elo_surface")
    form5 = safe("form_winrate_5")
    form10 = safe("form_winrate_10")
    days_since = safe("days_since_last")
    matches30 = safe("matches_last30")
    rank = safe("rank")

    return PlayerSnapshot(
        elo=elo,
        elo_surface=elo_surface,
        form5=form5,
        form10=form10,
        days_since_last=days_since,
        matches_last30=matches30,
        rank=rank,
    )


def compute_h2h(
    history: pd.DataFrame,
    playerA: str,
    playerB: str,
    date: pd.Timestamp,
) -> Tuple[int, float, float]:
    """
    Bu tarihe kadar A vs B H2H istatistikleri.
    history dataset'inde playerA = winner, playerB = loser olduğunu varsayıyoruz.
    """
    h = history[
        (
            ((history["playerA"] == playerA) & (history["playerB"] == playerB))
            | ((history["playerA"] == playerB) & (history["playerB"] == playerA))
        )
        & (history["date"] < date)
    ]

    total = len(h)
    if total == 0:
        return 0, np.nan, np.nan

    winsA = (h["playerA"] == playerA).sum()
    winsB = (h["playerA"] == playerB).sum()

    return total, winsA / total, winsB / total


def build_feature_row(
    history: pd.DataFrame,
    feature_cols: List[str],
    playerA: str,
    playerB: str,
    surface: str,
    date: pd.Timestamp,
    round_code: Optional[str] = None,
    oddsA: Optional[float] = None,
    oddsB: Optional[float] = None,
) -> pd.DataFrame:
    """
    Logistic regression modelinin beklediği feature kolonlarıyla uyumlu
    tek satırlık bir DataFrame üretir.
    Bilinmeyen feature'lar NaN bırakılır (imputer dolduracak).
    """
    row: Dict[str, float] = {c: np.nan for c in feature_cols}

    # 1) Player snapshot'ları
    snapA = get_player_snapshot(history, playerA, date, surface)
    snapB = get_player_snapshot(history, playerB, date, surface)

    if snapA:
        row["eloA"] = snapA.elo
        row["elo_surfaceA"] = snapA.elo_surface
        row["form_winrateA_5"] = snapA.form5
        row["form_winrateA_10"] = snapA.form10
        row["days_since_lastA_clipped"] = min(max(snapA.days_since_last, 0), 365)
        row["matches_last30A"] = snapA.matches_last30
        row["rankA"] = snapA.rank

    if snapB:
        row["eloB"] = snapB.elo
        row["elo_surfaceB"] = snapB.elo_surface
        row["form_winrateB_5"] = snapB.form5
        row["form_winrateB_10"] = snapB.form10
        row["days_since_lastB_clipped"] = min(max(snapB.days_since_last, 0), 365)
        row["matches_last30B"] = snapB.matches_last30
        row["rankB"] = snapB.rank

    # Fark feature'ları
    if "eloA" in row and "eloB" in row and not np.isnan(row["eloA"]) and not np.isnan(row["eloB"]):
        row["elo_diff"] = row["eloA"] - row["eloB"]

    if (
        "elo_surfaceA" in row
        and "elo_surfaceB" in row
        and not np.isnan(row["elo_surfaceA"])
        and not np.isnan(row["elo_surfaceB"])
    ):
        row["elo_surface_diff"] = row["elo_surfaceA"] - row["elo_surfaceB"]

    if (
        "form_winrateA_5" in row
        and "form_winrateB_5" in row
        and not np.isnan(row["form_winrateA_5"])
        and not np.isnan(row["form_winrateB_5"])
    ):
        row["form_winrate_diff_5"] = row["form_winrateA_5"] - row["form_winrateB_5"]

    if (
        "form_winrateA_10" in row
        and "form_winrateB_10" in row
        and not np.isnan(row["form_winrateA_10"])
        and not np.isnan(row["form_winrateB_10"])
    ):
        row["form_winrate_diff_10"] = row["form_winrateA_10"] - row["form_winrateB_10"]

    if "matches_last30A" in row and "matches_last30B" in row:
        if not np.isnan(row["matches_last30A"]) and not np.isnan(row["matches_last30B"]):
            row["matches_last30_diff"] = row["matches_last30A"] - row["matches_last30B"]

    if "rankA" in row and "rankB" in row:
        if not np.isnan(row["rankA"]) and not np.isnan(row["rankB"]):
            row["rank_diff"] = row["rankA"] - row["rankB"]

    if (
        "days_since_lastA_clipped" in row
        and "days_since_lastB_clipped" in row
        and not np.isnan(row["days_since_lastA_clipped"])
        and not np.isnan(row["days_since_lastB_clipped"])
    ):
        row["days_since_last_diff_clipped"] = (
            row["days_since_lastA_clipped"] - row["days_since_lastB_clipped"]
        )

    # 2) H2H feature'ları
    h2h_matches, h2hA, h2hB = compute_h2h(history, playerA, playerB, date)
    row["h2h_matches"] = h2h_matches
    row["h2h_winrateA"] = h2hA
    row["h2h_winrateB"] = h2hB
    if not np.isnan(h2hA) and not np.isnan(h2hB):
        row["h2h_winrate_diff"] = h2hA - h2hB

    # 3) Round feature'ları
    round_importance, is_final, is_semi, is_quarter = map_round_importance(round_code)
    row["round_importance"] = round_importance
    row["is_final"] = is_final
    row["is_semi"] = is_semi
    row["is_quarter"] = is_quarter

    # 4) Surface one-hot
    for c in feature_cols:
        if c.startswith("surface_"):
            surf_label = c.split("surface_", 1)[1]
            row[c] = 1.0 if surf_label.lower() == surface.lower() else 0.0

    # 5) Market / odds (opsiyonel)
    if oddsA is not None and oddsB is not None and oddsA > 0 and oddsB > 0:
        invA = 1.0 / oddsA
        invB = 1.0 / oddsB
        denom = invA + invB
        pA = invA / denom
        pB = invB / denom

        row["oddsA"] = oddsA
        row["oddsB"] = oddsB
        row["pA_market"] = pA
        row["pB_market"] = pB
        row["p_diff"] = pA - pB
        eps = 1e-6
        row["logit_pA_market"] = np.log((pA + eps) / (1.0 - pA + eps))

    # Tek satırlık DataFrame
    df_row = pd.DataFrame([row], columns=feature_cols)
    return df_row


def predict_single_match(
    playerA: str,
    playerB: str,
    surface: str,
    date_str: str,
    round_code: Optional[str] = None,
    oddsA: Optional[float] = None,
    oddsB: Optional[float] = None,
) -> float:
    """Verilen parametreler için p(A kazanır) döner."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"History dataset bulunamadı: {HISTORY_PATH}")

    history = pd.read_csv(HISTORY_PATH)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    date = pd.to_datetime(date_str)

    feature_cols = load_feature_list(FEATURE_LIST_PATH)
    model = load(MODEL_PATH)
    imputer = load(IMPUTER_PATH)

    X_row = build_feature_row(
        history=history,
        feature_cols=feature_cols,
        playerA=playerA,
        playerB=playerB,
        surface=surface,
        date=date,
        round_code=round_code,
        oddsA=oddsA,
        oddsB=oddsB,
    )

    X_imp = imputer.transform(X_row)
    p = model.predict_proba(X_imp)[0, 1]
    return float(p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="What-if: İki oyuncu için tek maç tahmini üretir."
    )
    parser.add_argument("--playerA", required=True, help="Oyuncu A ismi (bizim dataset'teki haliyle)")
    parser.add_argument("--playerB", required=True, help="Oyuncu B ismi")
    parser.add_argument("--surface", required=True, help="Zemin: Hard / Clay / Grass / Carpet")
    parser.add_argument("--date", required=True, help="Tarih (YYYY-MM-DD)")
    parser.add_argument(
        "--round",
        dest="round_code",
        default=None,
        help="Round kodu (F, SF, QF, R16, R32, ...)",
    )
    parser.add_argument("--oddsA", type=float, default=None, help="Opsiyonel: A için oran")
    parser.add_argument("--oddsB", type=float, default=None, help="Opsiyonel: B için oran")

    args = parser.parse_args()

    p = predict_single_match(
        playerA=args.playerA,
        playerB=args.playerB,
        surface=args.surface,
        date_str=args.date,
        round_code=args.round_code,
        oddsA=args.oddsA,
        oddsB=args.oddsB,
    )

    print(
        f"[whatif] {args.date} tarihinde, {args.surface} zeminde,\n"
        f"  {args.playerA} vs {args.playerB} için model olasılığı (A kazanır): {p:.4f}"
    )

    if args.oddsA and args.oddsB:
        invA = 1.0 / args.oddsA
        invB = 1.0 / args.oddsB
        denom = invA + invB
        p_market = invA / denom
        edge = p - p_market
        print(f"[whatif] Market implied p(A)= {p_market:.4f}, edge = {edge:+.4f}")


if __name__ == "__main__":
    main()
