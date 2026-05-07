"""
API-Tennis client with a tiny on-disk cache.

The cache keys on the request URL+params and stores the JSON payload in
``data/cache/api_tennis/`` for ``ttl_s`` seconds. Mostly useful during dev
and for repeated UI refreshes — production should still rely on the
caller's rate-limit hygiene.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils.config import DATA_DIR


CACHE_DIR = DATA_DIR / "cache" / "api_tennis"


@dataclass(frozen=True)
class ApiTennisConfig:
    api_key: str
    base_url: str = "https://api.api-tennis.com/tennis/"
    timeout_s: int = 30
    proxy: Optional[str] = None
    cache_ttl_s: int = 0  # 0 disables cache


def _cache_path(method: str, params: Dict[str, Any]) -> Path:
    safe_params = {k: v for k, v in params.items() if k.lower() != "apikey"}
    blob = json.dumps({"method": method, "params": safe_params}, sort_keys=True, default=str)
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{method}_{h}.json"


def _read_cache(path: Path, ttl_s: int) -> Optional[Dict[str, Any]]:
    if ttl_s <= 0 or not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_s:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Cache failures must never break the call.
        pass


def _get(cfg: ApiTennisConfig, params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(params.get("method", "unknown"))
    cache_path = _cache_path(method, params)
    cached = _read_cache(cache_path, cfg.cache_ttl_s)
    if cached is not None:
        return cached

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
    out = payload if isinstance(payload, dict) else {"result": payload}
    _write_cache(cache_path, out)
    return out


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


def consensus_decimal_moneyline(odds_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], int]:
    """
    Extract a consensus (median) Home/Away decimal price across books.

    Returns ``(home, away, n_books)`` where ``n_books`` counts books that
    contributed to the median. ``(None, None, 0)`` if nothing parseable.
    """
    if not isinstance(odds_payload, dict) or not odds_payload:
        return None, None, 0

    # Some calls return {match_key: {...}}, unwrap if so.
    root = odds_payload
    if len(root) == 1:
        first_key = next(iter(root.keys()))
        if first_key.isdigit() and isinstance(root[first_key], dict):
            root = root[first_key]

    ha = root.get("Home/Away") if isinstance(root, dict) else None
    if not isinstance(ha, dict):
        return None, None, 0

    home_d = ha.get("Home")
    away_d = ha.get("Away")
    if not isinstance(home_d, dict) or not isinstance(away_d, dict):
        return None, None, 0

    def _to_f(x: Any) -> Optional[float]:
        try:
            v = float(str(x).strip())
            return v if v > 1.0 else None
        except Exception:
            return None

    paired: list[tuple[float, float]] = []
    for book, hv in home_d.items():
        ho = _to_f(hv)
        ao = _to_f(away_d.get(book))
        if ho and ao:
            paired.append((ho, ao))

    if not paired:
        return None, None, 0

    homes = sorted(h for h, _ in paired)
    aways = sorted(a for _, a in paired)
    mid = len(homes) // 2
    if len(homes) % 2:
        h_med = homes[mid]
        a_med = aways[mid]
    else:
        h_med = 0.5 * (homes[mid - 1] + homes[mid])
        a_med = 0.5 * (aways[mid - 1] + aways[mid])

    return float(h_med), float(a_med), len(paired)
