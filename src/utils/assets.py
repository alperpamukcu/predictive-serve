from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Optional


def slugify(text: str) -> str:
    """
    Create a filesystem-safe slug for asset lookup.
    Note: We intentionally keep this simple and deterministic.
    """
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


@dataclass(frozen=True)
class AssetPaths:
    root: Path

    @property
    def players(self) -> Path:
        return self.root / "players"

    @property
    def tournaments(self) -> Path:
        return self.root / "tournaments"


def find_image(path_no_ext: Path) -> Optional[Path]:
    """
    Try common image extensions.
    """
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = path_no_ext.with_suffix(ext)
        if p.exists():
            return p
    return None

