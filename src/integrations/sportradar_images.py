from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class SportradarImagesConfig:
    api_key: str
    sport: str = "tennis"
    access: str = "t"  # t (trial) or p (production)
    provider: str = "getty"
    league: str = ""  # only needed for soccer / getty golf
    image_type: str = "headshots"
    timeout_s: int = 45

    @property
    def base_url(self) -> str:
        return "https://api.sportradar.com"

    def _path_base(self) -> str:
        # OpenAPI: /{sport}-images-{access}3/{provider}/{league}/{image_type}/...
        # Tennis: league is omitted as an empty path segment → `getty//headshots`.
        league = (self.league or "").strip()
        if league:
            return f"/{self.sport}-images-{self.access}3/{self.provider}/{league}/{self.image_type}"
        return f"/{self.sport}-images-{self.access}3/{self.provider}//{self.image_type}"

    def _path_base_with_empty_league(self) -> str:
        # Fallback when API strictly expects the league segment.
        return f"/{self.sport}-images-{self.access}3/{self.provider}//{self.image_type}"


def _get_json(url: str, api_key: str, timeout_s: int) -> Dict[str, Any]:
    resp = requests.get(
        url,
        headers={"x-api-key": api_key, "accept": "application/json"},
        timeout=timeout_s,
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_player_manifest(cfg: SportradarImagesConfig, year: int) -> Dict[str, Any]:
    """
    Player Manifest (yearly).
    Docs: /{sport}-images-{access_level}3/{provider}/{league}/{image_type}/players/{year}/manifest.{format}
    We try a couple URL patterns for league handling.
    """
    candidates = [
        f"{cfg.base_url}{cfg._path_base()}/players/{int(year)}/manifest.json",
        f"{cfg.base_url}{cfg._path_base_with_empty_league()}/players/{int(year)}/manifest.json",
    ]
    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            return _get_json(url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def iter_assets(manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Manifest may use "assetlist" as list of assets.
    assets = manifest.get("assetlist") or manifest.get("assets") or []
    if isinstance(assets, list):
        for a in assets:
            if isinstance(a, dict):
                yield a


def _norm_name(s: str) -> str:
    return " ".join(str(s).replace("\u00a0", " ").split()).strip().lower()


def build_player_image_index(
    manifest: Dict[str, Any],
) -> Dict[str, Tuple[str, str]]:
    """
    Returns mapping: normalized_player_name -> (asset_id, file_name)
    Picks a reasonable small image (250w-resize / 120x120 crop) when available, else original.
    """
    idx: Dict[str, Tuple[str, str]] = {}
    for a in iter_assets(manifest):
        asset_id = a.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            continue
        refs = a.get("refs") or []
        if not isinstance(refs, list):
            continue
        player_name = None
        is_primary = False
        for r in refs:
            if not isinstance(r, dict):
                continue
            if r.get("type") == "profile" and isinstance(r.get("name"), str):
                player_name = r["name"]
                is_primary = bool(r.get("primary", False))
                break
        if not player_name:
            continue

        # Choose file name from links
        file_name = None
        links = a.get("links") or []
        if isinstance(links, list):
            preferred = ["250w-resize.jpg", "120x120-crop.jpg", "90x90-crop.jpg", "original.jpg"]
            hrefs: List[str] = [l.get("href") for l in links if isinstance(l, dict) and isinstance(l.get("href"), str)]
            # href example: /headshots/players/<asset_id>/250w-resize.jpg
            for pref in preferred:
                for h in hrefs:
                    if h.endswith("/" + pref):
                        file_name = pref
                        break
                if file_name:
                    break
            if not file_name and hrefs:
                file_name = hrefs[0].split("/")[-1]

        if not file_name:
            continue

        key = _norm_name(player_name)
        # Prefer primary images when duplicates exist
        if key not in idx or is_primary:
            idx[key] = (asset_id, file_name)
    return idx


def download_player_image(
    cfg: SportradarImagesConfig,
    asset_id: str,
    file_name: str,
    out_path: Path,
) -> None:
    candidates = [
        f"{cfg.base_url}{cfg._path_base()}/players/{asset_id}/{file_name}",
        f"{cfg.base_url}{cfg._path_base_with_empty_league()}/players/{asset_id}/{file_name}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            resp = requests.get(
                url,
                headers={"x-api-key": cfg.api_key},
                timeout=cfg.timeout_s,
                allow_redirects=True,
            )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            return
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def fetch_logo_manifest(cfg: SportradarImagesConfig, year: str) -> Dict[str, Any]:
    """
    Logo Manifest.
    Docs: /{sport}-images-{access_level}3/{provider}/{league}/logos/{year}/manifest.{format}
    Note: In trial, year may be empty.
    """
    year_part = str(year)
    candidates = [
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}/{cfg.league}/logos/{year_part}/manifest.json",
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}//logos/{year_part}/manifest.json",
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}/{cfg.league}/logos/manifest.json",
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}//logos/manifest.json",
    ]
    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            return _get_json(url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def build_logo_index(manifest: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """
    Returns mapping: normalized_ref_name -> (asset_id, file_name)
    Prefers smaller logo sizes when present.
    """
    idx: Dict[str, Tuple[str, str]] = {}
    for a in iter_assets(manifest):
        asset_id = a.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            continue

        # Choose file name from links
        file_name = None
        links = a.get("links") or []
        if isinstance(links, list):
            preferred = ["h250-max-resize.png", "h250-max-resize.jpg", "h250-max-resize.jpeg", "original.png", "original.jpg", "original.jpeg"]
            hrefs: List[str] = [l.get("href") for l in links if isinstance(l, dict) and isinstance(l.get("href"), str)]
            for pref in preferred:
                for h in hrefs:
                    if h.endswith("/" + pref):
                        file_name = pref
                        break
                if file_name:
                    break
            if not file_name and hrefs:
                file_name = hrefs[0].split("/")[-1]
        if not file_name:
            continue

        refs = a.get("refs") or []
        if not isinstance(refs, list):
            continue
        for r in refs:
            if not isinstance(r, dict):
                continue
            nm = r.get("name")
            if isinstance(nm, str) and nm.strip():
                k = _norm_name(nm)
                if k not in idx:
                    idx[k] = (asset_id, file_name)
    return idx


def download_logo_image(cfg: SportradarImagesConfig, asset_id: str, file_name: str, out_path: Path) -> None:
    candidates = [
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}/{cfg.league}/logos/{asset_id}/{file_name}",
        f"{cfg.base_url}/{cfg.sport}-images-{cfg.access}3/{cfg.provider}//logos/{asset_id}/{file_name}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            resp = requests.get(
                url,
                headers={"x-api-key": cfg.api_key},
                timeout=cfg.timeout_s,
                allow_redirects=True,
            )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            return
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err

