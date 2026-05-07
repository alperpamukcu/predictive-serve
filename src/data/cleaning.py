# src/data/cleaning.py

from pathlib import Path
import pandas as pd

from src.utils.config import PROCESSED_DIR
from src.data.schema import MATCH_COLUMNS

RAW_MATCHES_PATH = PROCESSED_DIR / "matches_allyears.csv"
CLEAN_MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"


def load_raw_matches(path: Path = RAW_MATCHES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Drop rows missing critical fields
    df = df.dropna(subset=["date", "playerA", "playerB"])

    # 2) Sane year window (covers tennis-data.co.uk archive)
    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= 2000) & (df["year"] <= 2026)]

    # 3) Coerce odds, keep odds optional (filtering them throws away coverage
    #    and biases features that are computed before the filter, since we
    #    drop market columns from the production model anyway).
    for col in ["oddsA", "oddsB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    has_oddsA = df["oddsA"].notna()
    has_oddsB = df["oddsB"].notna()
    both_odds = has_oddsA & has_oddsB
    if both_odds.any():
        in_range = (
            (df["oddsA"] >= 1.01) & (df["oddsA"] <= 100.0) &
            (df["oddsB"] >= 1.01) & (df["oddsB"] <= 100.0)
        )
        df = df[(~both_odds) | in_range]

    # 4) Filter incomplete matches (retired, walkover, abandon)
    df["comment"] = df["comment"].fillna("").astype(str)
    bad_mask = (
        df["comment"].str.contains("ret", case=False, na=False)
        | df["comment"].str.contains("walk", case=False, na=False)
        | df["comment"].str.contains("abandon", case=False, na=False)
        | df["comment"].str.contains("def\\.", case=False, na=False, regex=True)
    )
    df = df[~bad_mask]

    # 5) Drop self-matches (data quality nuisance)
    df = df[df["playerA_norm"].astype(str) != df["playerB_norm"].astype(str)]

    # 6) Deduplicate identical (date, playerA, playerB, tourney) rows
    dedupe_keys = [c for c in ["date", "tourney", "playerA_norm", "playerB_norm", "round"] if c in df.columns]
    if dedupe_keys:
        df = df.drop_duplicates(subset=dedupe_keys, keep="first")

    # 7) Drop temp year column, restore canonical column order
    df = df.drop(columns=["year"])
    df = df[MATCH_COLUMNS]

    return df


def build_clean_matches(
    input_path: Path = RAW_MATCHES_PATH,
    output_path: Path = CLEAN_MATCHES_PATH,
) -> Path:
    df_raw = load_raw_matches(input_path)
    df_clean = clean_matches(df_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[cleaning] Saved clean matches to: {output_path} (rows={len(df_clean)})")
    return output_path


if __name__ == "__main__":
    build_clean_matches()
