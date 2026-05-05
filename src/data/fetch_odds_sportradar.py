from __future__ import annotations

import datetime as dt
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv
from src.integrations.sportradar_odds import (
    SportradarOddsConfig,
    extract_moneyline_from_schedule_event,
    iter_schedule_sport_events,
)

FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"


def _clean_name(x) -> str:
    return "" if x is None else " ".join(str(x).replace("\u00a0", " ").split()).strip()


def _norm_key(name: str) -> str:
    return _clean_name(name).lower()


def _pair_key(pa: str, pb: str) -> Tuple[str, str]:
    a = _norm_key(pa)
    b = _norm_key(pb)
    return (a, b) if a <= b else (b, a)


def _relax_tokens(name: str) -> str:
    s = _norm_key(name).replace(",", " ")
    toks = sorted([t for t in s.split() if t])
    return " ".join(toks)


def _pair_relaxed(pa: str, pb: str) -> Tuple[str, str]:
    a = _relax_tokens(pa)
    b = _relax_tokens(pb)
    return (a, b) if a <= b else (b, a)


def _attach_odds_by_name(
    pA: str,
    pB: str,
    home_odds: float,
    away_odds: float,
    home_name: Optional[str],
    away_name: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    if home_name is None or away_name is None:
        return None, None

    hn = _norm_key(home_name)
    an = _norm_key(away_name)
    xa = _norm_key(pA)
    xb = _norm_key(pB)

    if xa == hn and xb == an:
        return float(home_odds), float(away_odds)
    if xa == an and xb == hn:
        return float(away_odds), float(home_odds)

    hn2 = _relax_tokens(home_name)
    an2 = _relax_tokens(away_name)
    xa2 = _relax_tokens(pA)
    xb2 = _relax_tokens(pB)

    if xa2 == hn2 and xb2 == an2:
        return float(home_odds), float(away_odds)
    if xa2 == an2 and xb2 == hn2:
        return float(away_odds), float(home_odds)

    return None, None


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("SPORTRADAR_ODDS_API_KEY")
    if not api_key:
        print("[odds] INFO: SPORTRADAR_ODDS_API_KEY not set. Skipping odds enrichment.")
        return 0

    package_type = getenv("SPORTRADAR_ODDS_PACKAGE", "row") or "row"
    access = getenv("SPORTRADAR_ODDS_ACCESS", "t") or "t"
    lang = getenv("SPORTRADAR_ODDS_LANG", "en") or "en"
    odds_format = getenv("SPORTRADAR_ODDS_FORMAT", "eu") or "eu"
    sport_id = getenv("SPORTRADAR_ODDS_SPORT_ID", "sr:sport:5") or "sr:sport:5"

    cfg = SportradarOddsConfig(
        api_key=api_key,
        package_type=package_type,
        access=access,
        language=lang,
        odds_format=odds_format,
    )

    if not FIXTURES_PATH.exists():
        print(f"[odds] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    if "playerA" not in df.columns or "playerB" not in df.columns:
        print("[odds] ERROR: fixtures CSV must contain playerA and playerB.")
        return 2

    for c in ["playerA", "playerB"]:
        df[c] = df[c].map(_clean_name)

    if "oddsA" not in df.columns:
        df["oddsA"] = pd.NA
    if "oddsB" not in df.columns:
        df["oddsB"] = pd.NA

    if "date" not in df.columns:
        print("[odds] ERROR: fixtures CSV must contain date column.")
        return 2

    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"])

    odds_by_day: Dict[dt.date, Dict[Tuple[str, str], Tuple[float, float, Optional[str], Optional[str]]]] = {}

    days = sorted({pd.Timestamp(x).normalize().date() for x in df["_d"].tolist()})
    print(f"[odds] Loading Odds Comparison schedule for Tennis ({sport_id}) across {len(days)} day(s) ...")

    for day in days:
        ev_index: Dict[Tuple[str, str], Tuple[float, float, Optional[str], Optional[str]]] = {}
        try:
            for ev in iter_schedule_sport_events(cfg, sport_id, day):
                ho, ao, hm, aw = extract_moneyline_from_schedule_event(ev)
                if ho is None or ao is None or not hm or not aw:
                    continue
                tup = (float(ho), float(ao), hm, aw)
                ev_index[_pair_key(hm, aw)] = tup
                ev_index[_pair_relaxed(hm, aw)] = tup
        except Exception as e:
            print(f"[odds] WARNING: schedule fetch failed for {day}: {e}")

        odds_by_day[day] = ev_index
        time.sleep(0.35)

    updated = 0
    skipped_no_row = 0

    for idx, r in df.iterrows():
        d = pd.Timestamp(r["_d"]).date()
        pA = str(r["playerA"])
        pB = str(r["playerB"])
        if not pA or not pB:
            continue

        curA = r.get("oddsA")
        curB = r.get("oddsB")
        if pd.notna(curA) and pd.notna(curB):
            continue

        tbl = odds_by_day.get(d) or {}
        row = tbl.get(_pair_key(pA, pB)) or tbl.get(_pair_relaxed(pA, pB))
        if not row:
            skipped_no_row += 1
            continue

        ho, ao, hm, hw = row
        oddsA, oddsB = _attach_odds_by_name(pA, pB, ho, ao, hm, hw)
        if oddsA is None or oddsB is None:
            skipped_no_row += 1
            continue

        df.at[idx, "oddsA"] = oddsA
        df.at[idx, "oddsB"] = oddsB
        updated += 1

    df = df.drop(columns=["_d"], errors="ignore")
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_PATH, index=False)

    nonzero = df["oddsA"].notna() & df["oddsB"].notna()
    print(f"[odds] Rows with odds={int(nonzero.sum()):,} / {len(df):,} | newly_set={updated}")
    print(f"[odds] Saved: {FIXTURES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
