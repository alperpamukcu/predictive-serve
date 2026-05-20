from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

from src.integrations.api_tennis import ApiTennisConfig, get_fixtures
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
        surface = (
            _clean(ev.get("tournament_surface"))
            or guess_surface_from_tournament(tournament)
        )
        out.append(
            {
                "match_id": match_id,
                "date": date_str,
                "event_time": _clean(ev.get("event_time")),
                "tournament": tournament,
                "surface": surface,
                "round": _round_guess(ev),
                "playerA": pA,
                "playerB": pB,
                "oddsA": None,
                "oddsB": None,
                # match status + result for finished/live cards
                "status": _clean(ev.get("event_status")),
                "score": _clean(ev.get("event_final_result")),
                "live_game": _clean(ev.get("event_game_result")),
                "winner_side": _clean(ev.get("event_winner")),  # "First Player" / "Second Player"
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
    past_days = max(0, min(getenv_int("UPCOMING_PAST_DAYS", 2), 14))
    today = dt.date.today()
    start = today - dt.timedelta(days=past_days)
    stop = today + dt.timedelta(days=days - 1)

    cfg = ApiTennisConfig(api_key=api_key, base_url=base_url, proxy=proxy, cache_ttl_s=cache_ttl)
    print(f"[api-tennis] Fetching fixtures: {start}..{stop} (cache_ttl={cache_ttl}s)")

    # API-Tennis 500s on >14-day windows. Chunk into ~14-day slices.
    events: List[Dict[str, Any]] = []
    cur = start
    while cur <= stop:
        chunk_stop = min(cur + dt.timedelta(days=13), stop)
        try:
            events.extend(get_fixtures(cfg, date_start=cur, date_stop=chunk_stop))
        except Exception as e:
            print(f"[api-tennis] WARNING: chunk {cur}..{chunk_stop} failed: {e}")
        cur = chunk_stop + dt.timedelta(days=1)

    rows = to_fixtures_rows(events)
    if not rows:
        print("[api-tennis] WARNING: No fixtures returned.")
        return 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "playerA", "playerB", "surface"])

    # Name resolution against the historical roster happens downstream in
    # merge_recent_results.py via canonical_parts; nothing to apply here.

    df = df.drop_duplicates(subset=["match_id", "date", "playerA", "playerB"], keep="first")
    df = df.sort_values(["date", "tournament", "playerA", "playerB"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    surface_breakdown = df["surface"].value_counts().to_dict()
    print(f"[api-tennis] Saved: {OUT_PATH} (rows={len(df)}, surfaces={surface_breakdown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
