from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.utils.assets import slugify, find_image, AssetPaths
from src.utils.config import PROJECT_ROOT, PROCESSED_DIR


FIXTURES_PATH = PROCESSED_DIR / "fixtures_upcoming.csv"
ASSETS = AssetPaths(PROJECT_ROOT / "assets")


def _clean(x) -> str:
    return "" if x is None else " ".join(str(x).replace("\u00a0", " ").split()).strip()


def _download(url: str, out_path: Path, timeout_s: int = 30) -> bool:
    try:
        resp = requests.get(url, timeout=timeout_s, allow_redirects=True)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main() -> int:
    if not FIXTURES_PATH.exists():
        print(f"[api-tennis-images] ERROR: Fixtures not found: {FIXTURES_PATH}")
        return 2

    df = pd.read_csv(FIXTURES_PATH)
    # UI uses canonical-mapped player names; fixtures also keep *_raw.
    needed = [
        "playerA",
        "playerB",
        "playerA_raw",
        "playerB_raw",
        "playerA_logo_url",
        "playerB_logo_url",
    ]
    for c in needed:
        if c not in df.columns:
            print("[api-tennis-images] INFO: No logo URL columns found in fixtures. Skipping.")
            return 0

    downloaded = 0
    for _, r in df.iterrows():
        for player_col, raw_col, url_col in [
            ("playerA", "playerA_raw", "playerA_logo_url"),
            ("playerB", "playerB_raw", "playerB_logo_url"),
        ]:
            name = _clean(r.get(player_col))
            raw = _clean(r.get(raw_col))
            url = _clean(r.get(url_col))
            if not name or not url or not url.startswith("http"):
                continue

            # Download to both canonical and raw slugs (prevents mismatch in UI lookup).
            for nm in [name, raw]:
                if not nm:
                    continue
                slug = slugify(nm)
                if find_image(ASSETS.players / slug):
                    continue
                out_path = ASSETS.players / f"{slug}.jpg"
                if _download(url, out_path):
                    downloaded += 1
            time.sleep(0.15)

    print(f"[api-tennis-images] Done. downloaded={downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

