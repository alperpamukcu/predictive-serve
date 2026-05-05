from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def try_load_dotenv(project_root: Path) -> None:
    """
    Best-effort .env loader (no hard dependency at runtime).
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(dotenv_path=env_path, override=False)


def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def getenv_int(name: str, default: int) -> int:
    v = getenv(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)

