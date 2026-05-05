from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR


@dataclass(frozen=True)
class QualityReport:
    name: str
    rows: int
    date_min: Optional[str]
    date_max: Optional[str]
    missing: Dict[str, float]
    odds_coverage: float
    surfaces: Dict[str, int]
    rounds: Dict[str, int]


def _missing_rate(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(df)
    for c in cols:
        if c not in df.columns or n == 0:
            out[c] = 1.0
        else:
            out[c] = float(df[c].isna().mean())
    return out


def make_report(path: Path, name: str) -> QualityReport:
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        date_min = str(d.min().date()) if d.notna().any() else None
        date_max = str(d.max().date()) if d.notna().any() else None
    else:
        date_min = date_max = None

    key_cols = ["date", "tourney", "tournament", "surface", "round", "playerA", "playerB", "rankA", "rankB", "oddsA", "oddsB", "score", "comment"]
    missing = _missing_rate(df, key_cols)

    if {"oddsA", "oddsB"}.issubset(set(df.columns)):
        odds_coverage = float((pd.to_numeric(df["oddsA"], errors="coerce").notna() & pd.to_numeric(df["oddsB"], errors="coerce").notna()).mean())
    else:
        odds_coverage = 0.0

    surfaces = df["surface"].fillna("—").astype(str).value_counts().head(10).to_dict() if "surface" in df.columns else {}
    rounds = df["round"].fillna("—").astype(str).value_counts().head(10).to_dict() if "round" in df.columns else {}

    return QualityReport(
        name=name,
        rows=len(df),
        date_min=date_min,
        date_max=date_max,
        missing=missing,
        odds_coverage=odds_coverage,
        surfaces=surfaces,
        rounds=rounds,
    )


def main() -> None:
    candidates = [
        ("matches_allyears.csv", PROCESSED_DIR / "matches_allyears.csv"),
        ("matches_clean.csv", PROCESSED_DIR / "matches_clean.csv"),
        ("matches_with_elo_form_sets.csv", PROCESSED_DIR / "matches_with_elo_form_sets.csv"),
        ("train_dataset.csv", PROCESSED_DIR / "train_dataset.csv"),
        ("sportradar_historical.csv", (PROCESSED_DIR.parent / "raw" / "sportradar_historical.csv")),
    ]

    print("=== Data Quality Report ===")
    for name, path in candidates:
        if not path.exists():
            print(f"\n[{name}] MISSING: {path}")
            continue
        r = make_report(path, name)
        print(f"\n[{r.name}] rows={r.rows:,} date={r.date_min}..{r.date_max} odds_coverage={r.odds_coverage:.3f}")
        # show a few high-signal missing rates
        for k in ["date", "surface", "round", "playerA", "playerB", "rankA", "rankB", "oddsA", "oddsB", "score", "comment"]:
            if k in r.missing:
                print(f"  - missing[{k}]: {r.missing[k]:.3f}")
        if r.surfaces:
            print(f"  - top surfaces: {r.surfaces}")
        if r.rounds:
            print(f"  - top rounds: {r.rounds}")


if __name__ == '__main__':
    main()

