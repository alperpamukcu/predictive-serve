"""
Pull the most recent finished ATP singles matches from API-Tennis and
normalise them into our ``matches_clean.csv`` schema so they can be merged
into the historical dataset. This is how the live site stays current
between tennis-data.uk's weekly publishing cadence.

Output: ``data/processed/recent_results_apitennis.csv``

The merge into matches_clean.csv (and the cascade through elo / form /
features / training) is handled by ``src.data.merge_recent_results``.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.schema import MATCH_COLUMNS
from src.integrations.api_tennis import ApiTennisConfig, get_fixtures
from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import getenv, getenv_int, try_load_dotenv
from src.utils.surface import guess_surface_from_tournament

OUT_PATH = PROCESSED_DIR / "recent_results_apitennis.csv"

# Tournament categories that should never feed into the model. Lowercase
# substring match against the API tournament name + event_type fields.
EXCLUDE_KEYWORDS = (
    "challenger",
    "itf",
    "futures",
    "wta",          # ATP-only model
    "exhibition",
    "junior",
    "doubles",
    "wheelchair",
)


def _clean(x: Any) -> str:
    return "" if x is None else " ".join(str(x).replace(" ", " ").split()).strip()


def _is_doubles_name(name: str) -> bool:
    return "/" in name


def _is_atp_singles(ev: Dict[str, Any]) -> bool:
    """Filter that mirrors what we trained on (men's ATP singles)."""
    haystack = " ".join(
        (ev.get(k) or "")
        for k in ("event_type_type", "tournament_name")
    ).lower()
    if any(k in haystack for k in EXCLUDE_KEYWORDS):
        return False
    a = ev.get("event_first_player") or ""
    b = ev.get("event_second_player") or ""
    if _is_doubles_name(a) or _is_doubles_name(b):
        return False
    return True


def _parse_score(ev: Dict[str, Any], winner_is_first: bool) -> Dict[str, Optional[int]]:
    """Return W1/L1...W5/L5 columns from the API ``scores`` array.

    From the API winner's point of view: W = winner's set games,
    L = loser's set games. So if the score is "6-4 6-3" with the FIRST
    player winning, W1/L1 = 6/4, W2/L2 = 6/3.
    """
    out = {f"W{i}": None for i in range(1, 6)}
    out.update({f"L{i}": None for i in range(1, 6)})

    scores = ev.get("scores") or []
    if isinstance(scores, list):
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            set_num_raw = str(entry.get("set_number") or "").strip()
            m = re.search(r"(\d+)", set_num_raw)
            if not m:
                continue
            idx = int(m.group(1))
            if idx < 1 or idx > 5:
                continue
            try:
                s1 = int(entry.get("score_first") or 0)
                s2 = int(entry.get("score_second") or 0)
            except Exception:
                continue
            if winner_is_first:
                out[f"W{idx}"], out[f"L{idx}"] = s1, s2
            else:
                out[f"W{idx}"], out[f"L{idx}"] = s2, s1

    if not any(out[f"W{i}"] is not None for i in range(1, 6)):
        # Fallback: parse event_final_result like "6-4 6-3"
        final = (ev.get("event_final_result") or "").strip()
        if final:
            sets = re.findall(r"(\d+)\s*[-:]\s*(\d+)", final)
            for idx, (a, b) in enumerate(sets[:5], start=1):
                try:
                    a_i, b_i = int(a), int(b)
                except Exception:
                    continue
                if winner_is_first:
                    out[f"W{idx}"], out[f"L{idx}"] = a_i, b_i
                else:
                    out[f"W{idx}"], out[f"L{idx}"] = b_i, a_i
    return out


def _score_string(parsed: Dict[str, Optional[int]]) -> str:
    parts: List[str] = []
    for i in range(1, 6):
        w = parsed.get(f"W{i}")
        l = parsed.get(f"L{i}")
        if w is None or l is None:
            continue
        parts.append(f"{w}-{l}")
    return " ".join(parts)


def _is_finished(ev: Dict[str, Any]) -> bool:
    status = (ev.get("event_status") or "").lower()
    if any(k in status for k in ("finished", "ended", "walkover")):
        return True
    final = (ev.get("event_final_result") or "").strip()
    if final and (ev.get("event_winner") or "").strip():
        return True
    return False


def _to_matches_row(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _is_finished(ev) or not _is_atp_singles(ev):
        return None

    p_first = _clean(ev.get("event_first_player"))
    p_second = _clean(ev.get("event_second_player"))
    if not p_first or not p_second:
        return None

    winner_side = (ev.get("event_winner") or "").lower()
    if "first" in winner_side:
        winner_is_first = True
    elif "second" in winner_side:
        winner_is_first = False
    else:
        return None  # no clear winner

    playerA = p_first if winner_is_first else p_second  # winner
    playerB = p_second if winner_is_first else p_first  # loser

    date_str = _clean(ev.get("event_date"))
    if not date_str:
        return None
    tournament = _clean(ev.get("tournament_name") or ev.get("event_type_type") or "")
    surface = _clean(ev.get("tournament_surface")) or guess_surface_from_tournament(tournament)

    parsed = _parse_score(ev, winner_is_first=winner_is_first)
    score = _score_string(parsed)

    row: Dict[str, Any] = {
        "date": date_str,
        "tourney": tournament,
        "surface": surface,
        "round": _clean(ev.get("tournament_round")),
        "playerA": playerA,
        "playerB": playerB,
        "rankA": None,
        "rankB": None,
        "oddsA": None,
        "oddsB": None,
        "score": score,
        "comment": "Completed",
        "source_file": "api-tennis",
        "gender": "M",
        "winner": "A",
        "playerA_norm": playerA.strip().lower(),
        "playerB_norm": playerB.strip().lower(),
        "pA_implied_fair": None,
        "pB_implied_fair": None,
    }
    return row


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)
    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[recent-results] API_TENNIS_KEY missing — skipping.")
        return 0

    cfg = ApiTennisConfig(api_key=api_key, cache_ttl_s=getenv_int("API_TENNIS_CACHE_TTL_S", 600))
    past_days = max(1, min(getenv_int("RECENT_RESULTS_PAST_DAYS", 14), 60))
    today = dt.date.today()
    start = today - dt.timedelta(days=past_days)

    print(f"[recent-results] Fetching finished singles from {start}..{today}")

    # API rejects >14-day windows; chunk if necessary.
    events: List[Dict[str, Any]] = []
    cur = start
    while cur <= today:
        chunk_stop = min(cur + dt.timedelta(days=13), today)
        try:
            events.extend(get_fixtures(cfg, date_start=cur, date_stop=chunk_stop))
        except Exception as e:
            print(f"[recent-results] WARNING: chunk {cur}..{chunk_stop} failed: {e}")
        cur = chunk_stop + dt.timedelta(days=1)

    rows: List[Dict[str, Any]] = []
    rejected = {"not_finished": 0, "not_atp_singles": 0, "no_winner": 0, "missing_date": 0}
    for ev in events:
        if not _is_finished(ev):
            rejected["not_finished"] += 1
            continue
        if not _is_atp_singles(ev):
            rejected["not_atp_singles"] += 1
            continue
        row = _to_matches_row(ev)
        if row is None:
            rejected["no_winner"] += 1
            continue
        rows.append(row)

    if not rows:
        print(f"[recent-results] No qualifying matches. Rejections: {rejected}")
        return 0

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date", "playerA_norm", "playerB_norm"], keep="first")
    df = df.sort_values("date")
    df = df.reindex(columns=MATCH_COLUMNS, fill_value=None)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(
        f"[recent-results] Saved {len(df)} ATP singles to {OUT_PATH}. "
        f"Window {df['date'].min()} -> {df['date'].max()}. Rejections: {rejected}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
