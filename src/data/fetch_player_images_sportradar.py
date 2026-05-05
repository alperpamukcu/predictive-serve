from __future__ import annotations

import datetime as dt
from pathlib import Path
import time
import json

import pandas as pd

from src.utils.config import PROJECT_ROOT, PROCESSED_DIR
from src.utils.env import try_load_dotenv, getenv
from src.utils.assets import slugify, find_image, AssetPaths
from src.integrations.sportradar_images import (
    SportradarImagesConfig,
    fetch_player_manifest,
    build_player_image_index,
    download_player_image,
)


FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ASSETS = AssetPaths(PROJECT_ROOT / "assets")


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("SPORTRADAR_IMAGES_API_KEY")
    if not api_key:
        print("[images] INFO: SPORTRADAR_IMAGES_API_KEY not set. Skipping image download.")
        return 0

    sport = getenv("SPORTRADAR_IMAGES_SPORT", "tennis") or "tennis"
    access = getenv("SPORTRADAR_IMAGES_ACCESS", "t") or "t"
    provider = getenv("SPORTRADAR_IMAGES_PROVIDER", "getty") or "getty"
    league = getenv("SPORTRADAR_IMAGES_LEAGUE", "") or ""

    year_str = getenv("SPORTRADAR_IMAGES_MANIFEST_YEAR")
    year = int(year_str) if year_str and year_str.isdigit() else dt.date.today().year

    cfg = SportradarImagesConfig(
        api_key=api_key,
        sport=sport,
        access=access,
        provider=provider,
        league=league,
    )

    if not FIXTURES_PATH.exists():
        print(f"[images] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    players = []
    for c in ["playerA", "playerB"]:
        if c in df.columns:
            players.extend(df[c].dropna().astype(str).tolist())
    players = sorted({p.strip() for p in players if p and p.strip()})

    if not players:
        print("[images] No players found in fixtures.")
        return 0

    # Manifest cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / f"{sport}_{provider}_players_{year}_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        print(f"[images] Fetching player manifest {year} ({sport}/{provider}) ...")
        manifest = fetch_player_manifest(cfg, year=year)
        # Write raw JSON
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    idx = build_player_image_index(manifest)

    downloaded = 0
    missing = 0
    for p in players:
        slug = slugify(p)
        if find_image(ASSETS.players / slug):
            continue

        key = " ".join(p.split()).strip().lower()
        match = idx.get(key)
        if not match:
            missing += 1
            continue
        asset_id, file_name = match

        out_path = ASSETS.players / f"{slug}.jpg"
        try:
            download_player_image(cfg, asset_id=asset_id, file_name=file_name, out_path=out_path)
            downloaded += 1
        except Exception as e:
            print(f"[images] WARNING: {p} download failed: {e}")
        finally:
            time.sleep(0.2)

    print(f"[images] Done. downloaded={downloaded} missing_in_manifest={missing} total_players={len(players)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

