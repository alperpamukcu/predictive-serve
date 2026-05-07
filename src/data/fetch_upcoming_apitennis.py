from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

from src.integrations.api_tennis import ApiTennisConfig, get_fixtures
from src.utils.aliases import load_aliases
from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import getenv, getenv_int, try_load_dotenv
from src.utils.surface import guess_surface_from_tournament


OUT_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"


def _clean(x: Any) -> str:
    return "" if x is None else " ".join(str(x).replace(" ", " ").split()).strip()


def _round_guess(ev: Dict[str, Any]) -> str:
    return _clean(ev.get("tournament_round"))


def _tournament_name(ev: Dict[str, Any]) -> str:
    return _clean(ev.get("tournament_name") or ev.get("event_type_type") or "")


def _event_date(ev: Dict[str, Any]) -> Optional[str]:
    d = _clean(ev.get("event_date"))
    return d or None


def to_fixtures_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        match_id = _clean(ev.get("event_key") or ev.get("match_key") or "")
        date_str = _event_date(ev)
        pA = _clean(ev.get("event_first_player"))
        pB = _clean(ev.get("event_second_player"))
        if not date_str or not pA or not pB:
            continue
        tournament = _tournament_name(ev)
        # API-Tennis fixtures don't carry a reliable surface; infer from the
        # tournament name and fall back to "Hard" only as a last resort.
        surface = (
            _clean(ev.get("tournament_surface"))
            or guess_surface_from_tournament(tournament)
        )
        out.append(
            {
                "match_id": match_id,
                "date": date_str,
                "tournament": tournament,
                "surface": surface,
                "round": _round_guess(ev),
                "playerA": pA,
                "playerB": pB,
                "oddsA": None,
                "oddsB": None,
                # raw fields for later syncing/assets
                "playerA_raw": pA,
                "playerB_raw": pB,
                "tournament_raw": tournament,
                "playerA_logo_url": _clean(ev.get("event_first_player_logo")),
                "playerB_logo_url": _clean(ev.get("event_second_player_logo")),
            }
        )
    return out


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[api-tennis] ERROR: API_TENNIS_KEY is not set.")
        return 2

    base_url = getenv("API_TENNIS_BASE_URL", "https://api.api-tennis.com/tennis/") or "https://api.api-tennis.com/tennis/"
    proxy = getenv("API_TENNIS_PROXY")
    cache_ttl = getenv_int("API_TENNIS_CACHE_TTL_S", 600)

    days = max(1, min(getenv_int("UPCOMING_DAYS", 14), 30))
    start = dt.date.today()
    stop = start + dt.timedelta(days=days - 1)

    cfg = ApiTennisConfig(api_key=api_key, base_url=base_url, proxy=proxy, cache_ttl_s=cache_ttl)
    print(f"[api-tennis] Fetching fixtures: {start}..{stop} (cache_ttl={cache_ttl}s)")

    events = get_fixtures(cfg, date_start=start, date_stop=stop)
    rows = to_fixtures_rows(events)
    if not rows:
        print("[api-tennis] WARNING: No fixtures returned.")
        return 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "playerA", "playerB", "surface"])

    aliases = load_aliases()
    df["playerA"] = df["playerA"].map(aliases.map_player)
    df["playerB"] = df["playerB"].map(aliases.map_player)
    df["tournament"] = df["tournament"].map(aliases.map_tournament)

    df = df.drop_duplicates(subset=["match_id", "date", "playerA", "playerB"], keep="first")
    df = df.sort_values(["date", "tournament", "playerA", "playerB"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    surface_breakdown = df["surface"].value_counts().to_dict()
    print(f"[api-tennis] Saved: {OUT_PATH} (rows={len(df)}, surfaces={surface_breakdown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
