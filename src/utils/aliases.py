from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.config import PROJECT_ROOT


def _clean(s: str) -> str:
    return " ".join(str(s).replace("\u00a0", " ").split()).strip()


def _key(s: str) -> str:
    # keying is case-insensitive, display is not
    return _clean(s).lower()


@dataclass(frozen=True)
class Aliases:
    player: Dict[str, str]
    tournament: Dict[str, str]

    def map_player(self, name: str) -> str:
        k = _key(name)
        return self.player.get(k, _clean(name))

    def map_tournament(self, name: str) -> str:
        k = _key(name)
        return self.tournament.get(k, _clean(name))


def _load_alias_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "raw" not in df.columns or "canonical" not in df.columns:
        return {}
    m: Dict[str, str] = {}
    for _, r in df.iterrows():
        raw = r.get("raw")
        can = r.get("canonical")
        if not isinstance(raw, str) or not isinstance(can, str):
            continue
        raw_k = _key(raw)
        can_v = _clean(can)
        if raw_k and can_v and not raw_k.startswith("#") and not can_v.startswith("#"):
            m[raw_k] = can_v
    return m


def load_aliases() -> Aliases:
    registry = PROJECT_ROOT / "data" / "registry"
    return Aliases(
        player=_load_alias_file(registry / "player_aliases.csv"),
        tournament=_load_alias_file(registry / "tournament_aliases.csv"),
    )

