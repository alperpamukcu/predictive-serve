"""
Backfill player photos for every cached entry that has a ``player_key``
but no local image yet. Runs as part of the daily-refresh workflow so the
Players tab is fully populated without anyone clicking buttons.

Strategy:
    1. Read data/cache/player_meta.json
    2. For each entry with player_key + no local file under
       assets/players/<slug>.{jpg,png,...}, call get_players(key) once,
       grab player_logo, download to assets/players/<slug>.jpg, and persist
       any newly seen birthday into the cache.
    3. Stop early after MAX_PER_RUN downloads to keep the cron predictable.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import requests

from src.integrations.api_tennis import ApiTennisConfig, get_players
from src.utils.assets import find_image, slugify
from src.utils.config import DATA_DIR, PROJECT_ROOT
from src.utils.env import getenv, getenv_int, try_load_dotenv
from src.utils.player_meta import age_from_bday, load_cache, save_cache

ASSETS_PLAYERS = PROJECT_ROOT / "assets" / "players"
MAX_PER_RUN = 400  # respect API rate limits; the cron picks up the rest tomorrow


def _has_local_photo(name: str) -> bool:
    return find_image(ASSETS_PLAYERS / slugify(name)) is not None


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)
    api_key = getenv("API_TENNIS_KEY")
    if not api_key:
        print("[player-photos] API_TENNIS_KEY missing — skipping.")
        return 0

    cfg = ApiTennisConfig(api_key=api_key, cache_ttl_s=86400)
    cache = load_cache(DATA_DIR / "cache")
    if not cache:
        print("[player-photos] no cached players yet — skipping.")
        return 0

    todo = [
        (name, meta)
        for name, meta in cache.items()
        if meta.player_key and not _has_local_photo(name)
    ]
    print(f"[player-photos] candidates: {len(todo)} (cap {MAX_PER_RUN})")

    cap = getenv_int("PLAYER_PHOTOS_MAX_PER_RUN", MAX_PER_RUN)
    todo = todo[:cap]

    downloaded = 0
    enriched = 0
    for name, meta in todo:
        try:
            record = get_players(cfg, meta.player_key)
        except Exception as e:
            print(f"[player-photos] get_players failed for {name}: {e}")
            continue

        logo = record.get("player_logo") or meta.logo_url
        bday = record.get("player_bday") or meta.birthday
        if logo and not meta.logo_url:
            meta.logo_url = logo
            enriched += 1
        if bday and not meta.birthday:
            meta.birthday = bday
            meta.age = age_from_bday(bday)
            enriched += 1

        if logo and logo.startswith("http"):
            out = ASSETS_PLAYERS / f"{slugify(name)}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = requests.get(logo, timeout=20)
                r.raise_for_status()
                out.write_bytes(r.content)
                downloaded += 1
            except Exception as e:
                print(f"[player-photos] download failed for {name}: {e}")

    save_cache(DATA_DIR / "cache", cache)
    print(f"[player-photos] downloaded {downloaded}, enriched {enriched} cache rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
