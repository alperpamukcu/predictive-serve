from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.integrations.api_tennis import ApiTennisConfig, get_odds
from src.utils.config import PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv


FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _pick_best_decimal_moneyline(odds_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    API-Tennis returns nested dicts by market/book. We try to extract a reasonable decimal Home/Away pair.
    """
    if not odds_payload:
        return None, None

    # Common structure: result -> { match_key: { "Home/Away": { "Home": {book: odd}, "Away": {...}}}}
    # If called with a match_key, API may return { match_key: {...}} directly.
    root = odds_payload
    if len(root) == 1 and next(iter(root.keys())).isdigit():
        root = next(iter(root.values()))  # type: ignore

    ha = root.get("Home/Away") if isinstance(root, dict) else None
    if not isinstance(ha, dict):
        return None, None
    home = ha.get("Home")
    away = ha.get("Away")
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None, None

    # pick the first book with both odds parsable
    for book, hv in home.items():
        av = away.get(book)
        ho = _to_float(hv)
        ao = _to_float(av)
        if ho and ao and ho > 1.0 and ao > 1.0:
            return ho, ao

    # fallback: any parsable
    hs = [_to_float(v) for v in home.values()]
    as_ = [_to_float(v) for v in away.values()]
    hs = [x for x in hs if x and x > 1.0]
    as_ = [x for x in as_ if x and x > 1.0]
    if hs and as_:
        return float(hs[0]), float(as_[0])
    return None, None


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[api-tennis] INFO: API_TENNIS_KEY not set. Skipping odds enrichment.")
        return 0

    base_url = getenv("API_TENNIS_BASE_URL", "https://api.api-tennis.com/tennis/") or "https://api.api-tennis.com/tennis/"
    proxy = getenv("API_TENNIS_PROXY")
    cfg = ApiTennisConfig(api_key=api_key, base_url=base_url, proxy=proxy)

    if not FIXTURES_PATH.exists():
        print(f"[api-tennis] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    if "match_id" not in df.columns:
        print("[api-tennis] ERROR: fixtures_upcoming.csv must contain match_id (event_key).")
        return 2

    if "oddsA" not in df.columns:
        df["oddsA"] = pd.NA
    if "oddsB" not in df.columns:
        df["oddsB"] = pd.NA

    # Safety: avoid long runs on large fixture windows
    max_rows = getenv("API_TENNIS_ODDS_MAX_ROWS")
    if max_rows and str(max_rows).strip().isdigit():
        df = df.head(int(str(max_rows).strip())).copy()

    updated = 0
    for idx, r in df.iterrows():
        if pd.notna(r.get("oddsA")) and pd.notna(r.get("oddsB")):
            continue
        mk = str(r.get("match_id") or "").strip()
        if not mk:
            continue
        try:
            payload = get_odds(cfg, match_key=mk)
            # payload shape: {match_key: {...}} or {...}
            inner = payload.get(mk) if isinstance(payload, dict) and mk in payload else payload
            ho, ao = _pick_best_decimal_moneyline(inner if isinstance(inner, dict) else {})
            if ho is None or ao is None:
                continue
            # API doesn't specify player ordering; treat Home as playerA for UI purposes.
            df.at[idx, "oddsA"] = float(ho)
            df.at[idx, "oddsB"] = float(ao)
            updated += 1
        except Exception as e:
            print(f"[api-tennis] WARNING: odds failed for {mk}: {e}")
        finally:
            time.sleep(0.25)

    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_PATH, index=False)
    filled = int((df["oddsA"].notna() & df["oddsB"].notna()).sum())
    print(f"[api-tennis] Odds filled: {filled:,}/{len(df):,} (newly_set={updated})")
    print(f"[api-tennis] Saved: {FIXTURES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

