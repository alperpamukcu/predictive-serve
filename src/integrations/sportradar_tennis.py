from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import datetime as dt
import time

import requests
from requests.exceptions import HTTPError


@dataclass(frozen=True)
class SportradarTennisConfig:
    api_key: str
    access_level: str = "trial"
    language: str = "en"
    timeout_s: int = 30

    @property
    def base_url(self) -> str:
        # Docs: https://api.sportradar.com/tennis/{access_level}/v3/{language_code}/...
        return f"https://api.sportradar.com/tennis/{self.access_level}/v3/{self.language}"


def _get_json(url: str, api_key: str, timeout_s: int) -> Dict[str, Any]:
    resp = requests.get(url, headers={"x-api-key": api_key}, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def fetch_daily_summaries(cfg: SportradarTennisConfig, day: dt.date) -> Dict[str, Any]:
    # Schedules daily summaries feed:
    # /schedules/{date}/summaries.{format}
    date_str = day.strftime("%Y-%m-%d")
    url = f"{cfg.base_url}/schedules/{date_str}/summaries.json"
    return _get_json(url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)


def iter_sport_events_from_daily_summaries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Daily summaries return a `summaries` list; each item has `sport_event` (+ status/statistics).
    Older/alternate payloads may expose `sport_events` at the top level.
    """
    out: List[Dict[str, Any]] = []
    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        for item in summaries:
            if not isinstance(item, dict):
                continue
            se = item.get("sport_event")
            if isinstance(se, dict):
                out.append(se)
        return out

    legacy = payload.get("sport_events")
    if isinstance(legacy, list):
        for ev in legacy:
            if isinstance(ev, dict):
                out.append(ev)
    return out


def iter_upcoming_events(
    cfg: SportradarTennisConfig,
    start_day: dt.date,
    days: int,
) -> Iterable[Dict[str, Any]]:
    for i in range(days):
        day = start_day + dt.timedelta(days=i)

        # Trial accounts hit 429 easily; retry a couple times per day.
        last_err: Optional[Exception] = None
        backoff_s = 0.8
        for _attempt in range(3):
            try:
                payload = fetch_daily_summaries(cfg, day)
                last_err = None
                break
            except HTTPError as e:
                last_err = e
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                if status != 429:
                    raise

                # Honor Retry-After if present, else exponential backoff.
                ra = None
                if resp is not None:
                    ra = resp.headers.get("Retry-After")
                if ra and str(ra).strip().isdigit():
                    time.sleep(min(int(str(ra).strip()), 10))
                else:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 1.9, 8.0)
                continue
            except Exception as e:
                last_err = e
                break

        if last_err is not None:
            # Best-effort: skip this day instead of crashing the whole run.
            continue

        for ev in iter_sport_events_from_daily_summaries(payload):
            yield ev


def _safe_get(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _pick_surface(ev: Dict[str, Any]) -> Optional[str]:
    # Best effort: surface sometimes lives under tournament / sport_event_context.
    for path in [
        ("sport_event_context", "surface", "name"),
        ("sport_event_context", "surface"),
        ("tournament_round", "surface", "name"),
    ]:
        v = _safe_get(ev, *path)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict) and "name" in v and isinstance(v["name"], str):
            return v["name"].strip()
    return None


def _pick_round(ev: Dict[str, Any]) -> Optional[str]:
    tr = ev.get("tournament_round") or {}
    if isinstance(tr, dict):
        r = tr.get("name") or tr.get("type")
        if isinstance(r, str) and r.strip():
            return r.strip()
    ctx = ev.get("sport_event_context") or {}
    if isinstance(ctx, dict):
        rnd = ctx.get("round")
        if isinstance(rnd, dict):
            nm = rnd.get("name")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
            n = rnd.get("number")
            if n is not None and str(n).strip():
                return f"Round {n}"
    return None


def _pick_tournament(ev: Dict[str, Any]) -> Optional[str]:
    t = ev.get("tournament") or {}
    if isinstance(t, dict):
        name = t.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    ctx = ev.get("sport_event_context") or {}
    if isinstance(ctx, dict):
        comp = ctx.get("competition")
        if isinstance(comp, dict):
            nm = comp.get("name")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
        season = ctx.get("season")
        if isinstance(season, dict):
            nm = season.get("name")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
    return None


def _pick_start_time(ev: Dict[str, Any]) -> Optional[str]:
    # Many feeds provide scheduled as ISO timestamp string
    for k in ["scheduled", "start_time"]:
        v = ev.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_competitors(ev: Dict[str, Any]) -> Optional[List[str]]:
    comps = ev.get("competitors")
    if not isinstance(comps, list) or len(comps) < 2:
        return None
    names: List[str] = []
    for c in comps:
        if not isinstance(c, dict):
            continue
        nm = c.get("name")
        if isinstance(nm, str) and nm.strip():
            names.append(nm.strip())
    return names if len(names) >= 2 else None


def to_fixtures_rows(
    events: Iterable[Dict[str, Any]],
    default_surface: str = "Hard",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("id") if isinstance(ev.get("id"), str) else ""
        scheduled = _pick_start_time(ev)
        tourn = _pick_tournament(ev) or ""
        surface = _pick_surface(ev) or default_surface
        rnd = _pick_round(ev) or ""
        comps = _pick_competitors(ev)
        if not comps:
            continue

        # Use first two competitors as A/B (order is arbitrary but consistent enough for pre-match display)
        playerA, playerB = comps[0], comps[1]

        # Keep date as YYYY-MM-DD when scheduled is missing
        date_str = scheduled[:10] if scheduled else None
        out.append(
            {
                "match_id": event_id,
                "date": date_str,
                "tournament": tourn,
                "surface": surface,
                "round": rnd,
                "playerA": playerA,
                "playerB": playerB,
                "oddsA": None,
                "oddsB": None,
            }
        )
    return out

