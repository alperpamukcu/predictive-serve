from __future__ import annotations

import datetime as dt
from pathlib import Path
import shutil
import sys

import pandas as pd

from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv, getenv_int
from src.utils.aliases import load_aliases
from src.integrations.sportradar_tennis import (
    SportradarTennisConfig,
    iter_upcoming_events,
    to_fixtures_rows,
)


OUT_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
EXAMPLE_PATH = PROJECT_ROOT / "data" / "examples" / "fixtures_upcoming.csv"


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("SPORTRADAR_TENNIS_API_KEY")
    if not api_key:
        print("[fixtures] ERROR: SPORTRADAR_TENNIS_API_KEY is not set. Add it to .env or environment variables.")
        return 2

    access_level = getenv("SPORTRADAR_TENNIS_ACCESS_LEVEL", "trial") or "trial"
    lang = getenv("SPORTRADAR_TENNIS_LANG", "en") or "en"
    days = getenv_int("UPCOMING_DAYS", 14)
    days = max(1, min(days, 30))

    cfg = SportradarTennisConfig(api_key=api_key, access_level=access_level, language=lang)

    start_day = dt.date.today()
    print(f"[fixtures] Fetching upcoming tennis schedule: start={start_day} days={days} (access_level={access_level}, lang={lang})")

    events = iter_upcoming_events(cfg, start_day=start_day, days=days)
    rows = to_fixtures_rows(events)

    if not rows:
        print("[fixtures] WARNING: No events returned. Check your Sportradar package access and limits.")
        # Fallback to example so UI can still run.
        if EXAMPLE_PATH.exists():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXAMPLE_PATH, OUT_PATH)
            print(f"[fixtures] Fallback: copied example fixtures to {OUT_PATH}")
            return 0
        return 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "playerA", "playerB", "surface"])

    # Apply canonical mapping (best effort)
    aliases = load_aliases()
    df["playerA_raw"] = df["playerA"].astype(str)
    df["playerB_raw"] = df["playerB"].astype(str)
    df["tournament_raw"] = df["tournament"].astype(str)
    df["playerA"] = df["playerA"].map(aliases.map_player)
    df["playerB"] = df["playerB"].map(aliases.map_player)
    df["tournament"] = df["tournament"].map(aliases.map_tournament)

    df = df.drop_duplicates(subset=["match_id", "date", "playerA", "playerB"], keep="first")
    df = df.sort_values(["date", "tournament", "playerA", "playerB"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[fixtures] Saved: {OUT_PATH} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

