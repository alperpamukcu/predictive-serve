"""
Pull the ATP standings (~2k players) and merge them into the on-disk
player metadata cache. Designed to run from the daily-refresh workflow so
the Players tab has full coverage out of the box.

For each player not yet cached, we record:
    full_name, country, country_code, flag, player_key
Photos are downloaded lazily when the user opens that player's profile;
this script intentionally avoids hammering the API with thousands of
get_players calls.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from src.integrations.api_tennis import ApiTennisConfig, get_standings
from src.utils.config import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.utils.env import getenv, try_load_dotenv
from src.utils.player_meta import (
    PlayerMeta,
    attach_country,
    canonical_parts,
    load_cache,
    save_cache,
)


def _history_names() -> list[str]:
    p = PROCESSED_DIR / "matches_with_elo_form_sets.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, usecols=["playerA", "playerB"], low_memory=False).dropna()
    return pd.concat([df["playerA"], df["playerB"]]).astype(str).unique().tolist()


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)
    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[player-roster] API_TENNIS_KEY missing — skipping.")
        return 0

    cfg = ApiTennisConfig(api_key=api_key, cache_ttl_s=86400)
    try:
        rows = get_standings(cfg, "ATP")
    except Exception as e:
        print(f"[player-roster] get_standings failed: {e}")
        return 1

    print(f"[player-roster] standings rows: {len(rows)}")
    if not rows:
        return 0

    # Build a strict (initial, surname) index. We deliberately DO NOT fall
    # back to surname-only matches because that maps 'Murray A.' (Andy, who
    # retired and isn't in current ATP standings) to whichever surviving
    # 'Murray' surfaces first.
    by_init_sur: dict[tuple, dict] = {}
    for r in rows:
        full = (r.get("player") or "").strip()
        if not full:
            continue
        ci = canonical_parts(full)
        if not ci[1] or ci[0] is None:
            continue
        by_init_sur.setdefault(ci, r)

    cache_dir = DATA_DIR / "cache"
    cache = load_cache(cache_dir)
    history = _history_names()
    print(f"[player-roster] history names: {len(history)}, cached: {len(cache)}")

    new_resolved = 0
    for name in history:
        if name in cache and cache[name].fetched_at and not cache[name].not_found:
            continue
        i, sur = canonical_parts(name)
        if not sur or i is None:
            continue
        row = by_init_sur.get((i, sur))
        if not row:
            continue
        meta = PlayerMeta(name=name)
        meta.full_name = (row.get("player") or "").strip() or None
        meta.country = (row.get("country") or "").strip() or None
        try:
            meta.player_key = int(row.get("player_key")) if row.get("player_key") else None
        except Exception:
            meta.player_key = None
        meta.fetched_at = dt.datetime.utcnow().isoformat() + "Z"
        meta.not_found = meta.player_key is None
        cache[name] = attach_country(meta)
        new_resolved += 1

    save_cache(cache_dir, cache)
    print(f"[player-roster] resolved {new_resolved} new players (total cached: {len(cache)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
