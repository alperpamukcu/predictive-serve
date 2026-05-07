"""
Player metadata cache.

Resolves a tennis-data.uk style name (e.g. ``Sinner J.``) to API-Tennis
metadata: ``player_key``, ``full_name``, ``country``, ``country_code``,
``flag``, ``birthday``, ``logo_url``. Results are persisted on disk under
``data/cache/player_meta.json`` so subsequent calls are free.

The matcher uses a surname+initial canonical form so the API's
``J. Sinner`` lines up with the history's ``Sinner J.``.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from src.utils.country import country_code, flag_emoji


# ---------------------------------------------------------------------------
# Name canonicalisation
# ---------------------------------------------------------------------------

_DOUBLES_HINTS = ("/", " / ", " - ", " v ", " vs ")


def is_doubles(name: str) -> bool:
    """API fixtures sometimes carry doubles entries like ``Pereira/ Schiessl``."""
    if not name:
        return False
    return any(h in name for h in _DOUBLES_HINTS)


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def canonical_parts(name: str) -> tuple[Optional[str], str]:
    """Return ``(first_initial, surname)`` for *name* in lowercase.

    Both ``Sinner J.`` and ``J. Sinner`` come back as ``("j", "sinner")``.
    Hyphens collapse to spaces so ``Auger-Aliassime F.`` matches
    ``F. Auger-Aliassime``.
    """
    if not name:
        return None, ""
    s = _strip_accents(name).lower()
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"[-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p for p in s.split() if p]
    initial: Optional[str] = None
    surname_parts: list[str] = []
    for p in parts:
        if len(p) == 1:
            if initial is None:
                initial = p
        else:
            surname_parts.append(p)
    return initial, " ".join(surname_parts)


def names_match(a: str, b: str) -> bool:
    """Loose match on surname (and initial when both are known)."""
    ai, asur = canonical_parts(a)
    bi, bsur = canonical_parts(b)
    if not asur or not bsur:
        return False
    if asur == bsur or asur in bsur or bsur in asur:
        if ai and bi and ai != bi:
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

@dataclass
class PlayerMeta:
    name: str  # the history-format name we keyed on
    full_name: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    flag: Optional[str] = None
    birthday: Optional[str] = None  # 'DD.MM.YYYY' from API
    age: Optional[int] = None
    player_key: Optional[int] = None
    logo_url: Optional[str] = None
    fetched_at: Optional[str] = None
    not_found: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / "player_meta.json"


def load_cache(cache_dir: Path) -> dict[str, PlayerMeta]:
    p = _cache_path(cache_dir)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, PlayerMeta] = {}
    for k, v in raw.items():
        try:
            out[k] = PlayerMeta(**v)
        except Exception:
            continue
    return out


def save_cache(cache_dir: Path, cache: dict[str, PlayerMeta]) -> None:
    p = _cache_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: v.as_dict() for k, v in cache.items()}
    p.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _parse_birthday(bday: Optional[str]) -> Optional[dt.date]:
    if not bday:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(bday, fmt).date()
        except Exception:
            continue
    return None


def age_from_bday(bday: Optional[str], today: Optional[dt.date] = None) -> Optional[int]:
    d = _parse_birthday(bday)
    if d is None:
        return None
    today = today or dt.date.today()
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


def attach_country(meta: PlayerMeta) -> PlayerMeta:
    meta.country_code = country_code(meta.country)
    meta.flag = flag_emoji(meta.country)
    return meta


# ---------------------------------------------------------------------------
# Look-up against API-Tennis fixture payloads
# ---------------------------------------------------------------------------

def find_player_in_fixtures(
    name: str,
    fixtures: Iterable[dict[str, Any]],
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Scan ``fixtures`` (the payload from get_fixtures) for an entry whose
    first/second player matches *name* by surname + initial. Doubles entries
    are skipped.

    Returns ``(player_key, api_name, logo_url)``. Any field can be None.
    """
    for ev in fixtures:
        for nf, kf, lf in [
            ("event_first_player", "first_player_key", "event_first_player_logo"),
            ("event_second_player", "second_player_key", "event_second_player_logo"),
        ]:
            api_name = (ev.get(nf) or "").strip()
            if not api_name or is_doubles(api_name):
                continue
            if names_match(name, api_name):
                key = ev.get(kf)
                try:
                    key_int = int(key) if key is not None else None
                except Exception:
                    key_int = None
                logo = (ev.get(lf) or "").strip() or None
                return key_int, api_name, logo
    return None, None, None
