from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from requests.exceptions import HTTPError


@dataclass(frozen=True)
class SportradarOddsConfig:
    api_key: str
    package_type: str = "row"  # row or us
    access: str = "t"  # t (trial) or p (production)
    language: str = "en"
    odds_format: str = "eu"
    timeout_s: int = 35

    @property
    def base_url(self) -> str:
        return "https://api.sportradar.com"

    @property
    def path_prefix(self) -> str:
        # Example prefix: /oddscomparison-rowt1/en/eu
        return f"/oddscomparison-{self.package_type}{self.access}1/{self.language}/{self.odds_format}"


def fetch_daily_schedule_page(
    cfg: SportradarOddsConfig,
    sport_id: str,
    day: dt.date,
    start: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    ds = day.strftime("%Y-%m-%d")
    url = f"{cfg.base_url}{cfg.path_prefix}/sports/{sport_id}/{ds}/schedule.json"
    params: Dict[str, str] = {}
    if start:
        params["start"] = str(int(start))
    if limit:
        params["limit"] = str(max(1, min(int(limit), 100)))
    backoff = 0.7
    last_err: Optional[Exception] = None
    for _attempt in range(5):
        try:
            resp = requests.get(
                url,
                headers={"x-api-key": cfg.api_key, "accept": "application/json"},
                params=params or None,
                timeout=cfg.timeout_s,
            )

            if resp.status_code == 404:
                # No odds schedule published for that calendar day (common for far-future dates).
                return {"sport_events": []}

            resp.raise_for_status()
            return resp.json()
        except HTTPError as e:
            last_err = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                ra = None
                try:
                    ra = e.response.headers.get("Retry-After") if e.response is not None else None  # type: ignore
                except Exception:
                    ra = None
                if ra and str(ra).strip().isdigit():
                    time.sleep(min(int(str(ra).strip()), 15))
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.8, 10.0)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 10.0)

    if last_err:
        raise last_err
    return {"sport_events": []}


def iter_schedule_sport_events(
    cfg: SportradarOddsConfig,
    sport_id: str,
    day: dt.date,
    *,
    limit: int = 100,
    sleep_s: float = 0.35,
    max_pages: int = 50,
) -> Iterable[Dict[str, Any]]:
    start = 0
    pages = 0
    while pages < int(max_pages):
        payload = fetch_daily_schedule_page(cfg, sport_id, day, start=start, limit=limit)
        evs = payload.get("sport_events") or []
        if not isinstance(evs, list):
            return
        for ev in evs:
            if isinstance(ev, dict):
                yield ev
        if len(evs) < limit:
            return
        start += limit
        pages += 1
        time.sleep(float(sleep_s))


def _to_float_odds(odds: Any) -> Optional[float]:
    if odds is None:
        return None
    if isinstance(odds, (int, float)):
        return float(odds)
    s = str(odds).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def extract_moneyline_from_schedule_event(ev: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    home_name = away_name = None
    comps = ev.get("competitors") or []
    if isinstance(comps, list):
        for c in comps:
            if not isinstance(c, dict):
                continue
            qual = c.get("qualifier")
            nm = c.get("name")
            if not isinstance(nm, str):
                continue
            if qual == "home":
                home_name = nm
            elif qual == "away":
                away_name = nm

    consensus = ev.get("consensus") or {}
    lines = consensus.get("lines") or []
    if not isinstance(lines, list):
        return None, None, home_name, away_name

    target = None
    for ln in lines:
        if isinstance(ln, dict) and ln.get("name") in {"moneyline_current", "match_winner_current"}:
            target = ln
            break
    if target is None:
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            outs = ln.get("outcomes")
            if isinstance(outs, list) and outs and any(isinstance(o, dict) and o.get("type") in {"home", "away"} for o in outs):
                target = ln
                break

    if not isinstance(target, dict):
        return None, None, home_name, away_name

    home_odds = away_odds = None
    for o in target.get("outcomes") or []:
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t == "home":
            home_odds = _to_float_odds(o.get("odds"))
        elif t == "away":
            away_odds = _to_float_odds(o.get("odds"))

    return home_odds, away_odds, home_name, away_name


def fetch_sport_event_markets(cfg: SportradarOddsConfig, sport_event_id: str) -> Dict[str, Any]:
    url = f"{cfg.base_url}{cfg.path_prefix}/sport_events/{sport_event_id}/markets.json"
    resp = requests.get(url, headers={"x-api-key": cfg.api_key}, timeout=cfg.timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_consensus_moneyline_decimal(
    payload: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Markets endpoint variant (`sport_event` wrapper). Compatibility helper."""
    se = payload.get("sport_event") or {}

    home_name = away_name = None
    comps = se.get("competitors") or []
    if isinstance(comps, list):
        for c in comps:
            if not isinstance(c, dict):
                continue
            qual = c.get("qualifier")
            nm = c.get("name")
            if not isinstance(nm, str):
                continue
            if qual == "home":
                home_name = nm
            elif qual == "away":
                away_name = nm

    consensus = se.get("consensus") or {}
    lines = consensus.get("lines") or []
    if not isinstance(lines, list):
        return None, None, home_name, away_name

    target = None
    for ln in lines:
        if isinstance(ln, dict) and (ln.get("name") in {"moneyline_current", "match_winner_current"}):
            target = ln
            break
    if target is None:
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            outs = ln.get("outcomes")
            if isinstance(outs, list) and outs and any(isinstance(o, dict) and o.get("type") in {"home", "away"} for o in outs):
                target = ln
                break

    if not isinstance(target, dict):
        return None, None, home_name, away_name

    home_odds = away_odds = None
    for o in target.get("outcomes") or []:
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t == "home":
            home_odds = _to_float_odds(o.get("odds"))
        elif t == "away":
            away_odds = _to_float_odds(o.get("odds"))

    return home_odds, away_odds, home_name, away_name
