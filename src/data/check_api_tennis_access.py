from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import requests

from src.integrations.api_tennis import ApiTennisConfig, get_fixtures
from src.utils.config import PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv
import datetime as dt


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    status_code: Optional[int]
    message: str


def check() -> CheckResult:
    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        return CheckResult(False, None, "API_TENNIS_KEY not set.")

    base_url = getenv("API_TENNIS_BASE_URL", "https://api.api-tennis.com/tennis/") or "https://api.api-tennis.com/tennis/"
    proxy = getenv("API_TENNIS_PROXY")

    cfg = ApiTennisConfig(api_key=api_key, base_url=base_url, proxy=proxy)
    try:
        today = dt.date.today()
        _ = get_fixtures(cfg, today, today)
        return CheckResult(True, 200, f"OK: reachable ({base_url})" + (f" via proxy={proxy}" if proxy else ""))
    except requests.exceptions.SSLError as e:
        return CheckResult(False, None, f"SSL error. This often indicates ISP filtering. Use VPN or set API_TENNIS_PROXY. Details: {e}")
    except requests.exceptions.ProxyError as e:
        return CheckResult(False, None, f"Proxy error. Check API_TENNIS_PROXY. Details: {e}")
    except Exception as e:
        return CheckResult(False, None, f"Request failed: {e}")


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)
    res = check()
    if res.ok:
        print(f"[api-tennis] {res.message}")
        return 0
    print(f"[api-tennis] ERROR: {res.message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

