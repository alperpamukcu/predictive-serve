from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd

from src.utils.config import PROJECT_ROOT, PROCESSED_DIR
from src.utils.env import try_load_dotenv, getenv
from src.utils.assets import slugify, find_image, AssetPaths
from src.integrations.sportradar_images import (
    SportradarImagesConfig,
    fetch_logo_manifest,
    build_logo_index,
    download_logo_image,
)
from src.utils.aliases import load_aliases


FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ASSETS = AssetPaths(PROJECT_ROOT / "assets")


def main() -> int:
    try_load_dotenv(PROJECT_ROOT)

    api_key = getenv("SPORTRADAR_IMAGES_API_KEY")
    if not api_key:
        print("[logos] INFO: SPORTRADAR_IMAGES_API_KEY not set. Skipping logo download.")
        return 0

    # For logos, provider availability can differ; AP is common.
    sport = getenv("SPORTRADAR_IMAGES_SPORT", "tennis") or "tennis"
    access = getenv("SPORTRADAR_IMAGES_ACCESS", "t") or "t"
    provider = getenv("SPORTRADAR_LOGOS_PROVIDER", getenv("SPORTRADAR_IMAGES_PROVIDER", "ap") or "ap") or "ap"
    league = getenv("SPORTRADAR_IMAGES_LEAGUE", "") or ""

    year_str = getenv("SPORTRADAR_LOGOS_MANIFEST_YEAR", getenv("SPORTRADAR_IMAGES_MANIFEST_YEAR", ""))
    if access == "t":
        year = ""  # trial path often omits year
    else:
        year = year_str if year_str else str(dt.date.today().year)

    cfg = SportradarImagesConfig(
        api_key=api_key,
        sport=sport,
        access=access,
        provider=provider,
        league=league,
        image_type="headshots",  # unused for logos
    )

    if not FIXTURES_PATH.exists():
        print(f"[logos] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    if "tournament" not in df.columns:
        print("[logos] No tournament column found in fixtures.")
        return 0

    aliases = load_aliases()
    tournaments = sorted({aliases.map_tournament(t) for t in df["tournament"].dropna().astype(str).tolist() if t.strip()})
    if not tournaments:
        print("[logos] No tournaments in fixtures.")
        return 0

    # Manifest cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / f"{sport}_{provider}_logos_{access}_{year or 'trial'}_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        print(f"[logos] Fetching logo manifest (sport={sport}, provider={provider}, access={access}, year={year or 'trial'}) ...")
        manifest = fetch_logo_manifest(cfg, year=year)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    idx = build_logo_index(manifest)

    downloaded = 0
    missing = 0
    for tname in tournaments:
        slug = slugify(tname)
        if find_image(ASSETS.tournaments / slug):
            continue

        key = " ".join(tname.split()).strip().lower()
        match = idx.get(key)
        if not match:
            missing += 1
            continue
        asset_id, file_name = match

        ext = "." + file_name.split(".")[-1].lower()
        out_path = ASSETS.tournaments / f"{slug}{ext}"
        try:
            download_logo_image(cfg, asset_id=asset_id, file_name=file_name, out_path=out_path)
            downloaded += 1
        except Exception as e:
            print(f"[logos] WARNING: {tname} download failed: {e}")
        finally:
            time.sleep(0.2)

    print(f"[logos] Done. downloaded={downloaded} missing_in_manifest={missing} total_tournaments={len(tournaments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

