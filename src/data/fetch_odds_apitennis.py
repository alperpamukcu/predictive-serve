from __future__ import annotations

import time
from typing import Any, Dict

import pandas as pd

from src.integrations.api_tennis import ApiTennisConfig, consensus_decimal_moneyline, get_odds
from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import getenv, getenv_int, try_load_dotenv

FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[api-tennis] INFO: API_TENNIS_KEY not set. Skipping odds enrichment.")
        return 0

    base_url = getenv("API_TENNIS_BASE_URL", "https://api.api-tennis.com/tennis/") or "https://api.api-tennis.com/tennis/"
    proxy = getenv("API_TENNIS_PROXY")
    cache_ttl = getenv_int("API_TENNIS_ODDS_CACHE_TTL_S", 300)
    cfg = ApiTennisConfig(api_key=api_key, base_url=base_url, proxy=proxy, cache_ttl_s=cache_ttl)

    if not FIXTURES_PATH.exists():
        print(f"[api-tennis] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    if "match_id" not in df.columns:
        print("[api-tennis] ERROR: fixtures_upcoming.csv must contain match_id (event_key).")
        return 2

    for col in ("oddsA", "oddsB"):
        if col not in df.columns:
            df[col] = pd.NA
    if "n_books" not in df.columns:
        df["n_books"] = pd.NA

    max_rows = getenv("API_TENNIS_ODDS_MAX_ROWS")
    if max_rows and str(max_rows).strip().isdigit():
        df = df.head(int(str(max_rows).strip())).copy()

    sleep_s = max(0.05, float(getenv_int("API_TENNIS_ODDS_SLEEP_MS", 250)) / 1000.0)

    updated = 0
    book_counts: list[int] = []
    for idx, r in df.iterrows():
        if pd.notna(r.get("oddsA")) and pd.notna(r.get("oddsB")):
            continue
        mk = str(r.get("match_id") or "").strip()
        if not mk:
            continue
        try:
            payload: Dict[str, Any] = get_odds(cfg, match_key=mk)
            inner = payload.get(mk) if isinstance(payload, dict) and mk in payload else payload
            ho, ao, n = consensus_decimal_moneyline(inner if isinstance(inner, dict) else {})
            if ho is None or ao is None:
                continue
            df.at[idx, "oddsA"] = float(ho)
            df.at[idx, "oddsB"] = float(ao)
            df.at[idx, "n_books"] = int(n)
            book_counts.append(int(n))
            updated += 1
        except Exception as e:
            print(f"[api-tennis] WARNING: odds failed for {mk}: {e}")
        finally:
            time.sleep(sleep_s)

    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_PATH, index=False)
    filled = int((df["oddsA"].notna() & df["oddsB"].notna()).sum())
    avg_books = (sum(book_counts) / len(book_counts)) if book_counts else 0.0
    print(
        f"[api-tennis] Odds filled: {filled:,}/{len(df):,} (newly_set={updated}, avg_books={avg_books:.1f})"
    )
    print(f"[api-tennis] Saved: {FIXTURES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
