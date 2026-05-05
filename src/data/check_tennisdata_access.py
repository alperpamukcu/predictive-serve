from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import requests

from src.utils.config import PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv, getenv_int


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    status_code: Optional[int]
    message: str


def _proxies() -> dict | None:
    p = getenv("TEN_DATA_PROXY")
    if not p:
        return None
    return {"http": p, "https": p}


def check() -> CheckResult:
    base_url = getenv("TEN_DATA_BASE_URL", "http://www.tennis-data.co.uk") or "http://www.tennis-data.co.uk"
    timeout_s = int(getenv_int("TEN_DATA_TIMEOUT_S", 30))

    # probe one season file; use current year path as representative
    probe_year = getenv("TEN_DATA_PROBE_YEAR")
    if probe_year and probe_year.isdigit():
        year = int(probe_year)
    else:
        year = 2026

    url = f"{base_url.rstrip('/')}/{year}/{year}.xlsx"

    try:
        resp = requests.get(url, timeout=timeout_s, proxies=_proxies(), allow_redirects=True, stream=True)
        code = resp.status_code
        if code == 200:
            # Ensure it looks like an Excel file
            ct = (resp.headers.get("content-type") or "").lower()
            if "excel" in ct or "spreadsheet" in ct or url.endswith(".xlsx"):
                return CheckResult(True, code, f"OK: fetched {url} (content-type={ct or 'unknown'})")
            return CheckResult(True, code, f"OK: fetched {url} (content-type={ct or 'unknown'})")
        if code in (401, 403):
            return CheckResult(False, code, f"Blocked (HTTP {code}) for {url}. This usually means geo/IP restriction. Configure TEN_DATA_PROXY.")
        if code == 404:
            return CheckResult(False, code, f"Not found (HTTP 404) for {url}. Try a different TEN_DATA_PROBE_YEAR.")
        return CheckResult(False, code, f"Unexpected status HTTP {code} for {url}.")
    except requests.exceptions.ProxyError as e:
        return CheckResult(False, None, f"Proxy error. Check TEN_DATA_PROXY format/credentials. Details: {e}")
    except requests.exceptions.ConnectTimeout:
        return CheckResult(False, None, f"Connection timed out (timeout={timeout_s}s). Check proxy/network.")
    except Exception as e:
        return CheckResult(False, None, f"Request failed: {e}")


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)
    res = check()
    if res.ok:
        print(f"[tennis-data] {res.message}")
        return 0
    print(f"[tennis-data] ERROR: {res.message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

