from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class ApiTennisConfig:
    api_key: str
    base_url: str = "https://api.api-tennis.com/tennis/"
    timeout_s: int = 30
    proxy: Optional[str] = None


def _get(cfg: ApiTennisConfig, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    p["APIkey"] = cfg.api_key
    proxies = None
    if cfg.proxy:
        proxies = {"http": cfg.proxy, "https": cfg.proxy}
    resp = requests.get(cfg.base_url, params=p, timeout=cfg.timeout_s, proxies=proxies)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and payload.get("success") in (0, "0"):
        raise RuntimeError(str(payload))
    return payload if isinstance(payload, dict) else {"result": payload}


def get_fixtures(
    cfg: ApiTennisConfig,
    date_start: dt.date,
    date_stop: dt.date,
    *,
    event_type_key: Optional[str] = None,
    tournament_key: Optional[str] = None,
    timezone: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "method": "get_fixtures",
        "date_start": date_start.strftime("%Y-%m-%d"),
        "date_stop": date_stop.strftime("%Y-%m-%d"),
    }
    if event_type_key:
        params["event_type_key"] = event_type_key
    if tournament_key:
        params["tournament_key"] = tournament_key
    if timezone:
        params["timezone"] = timezone
    payload = _get(cfg, params)
    res = payload.get("result") or []
    return res if isinstance(res, list) else []


def get_odds(cfg: ApiTennisConfig, match_key: str) -> Dict[str, Any]:
    payload = _get(cfg, {"method": "get_odds", "match_key": str(match_key)})
    res = payload.get("result") or {}
    return res if isinstance(res, dict) else {}

