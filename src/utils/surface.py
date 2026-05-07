"""
Surface inference for fixtures whose API payload doesn't carry surface info.

API-Tennis fixtures rarely include a reliable surface field, but the model
needs one to populate the surface_* one-hot features. This module guesses
surface from tournament names with a curated keyword map and falls back
to "Hard" when nothing matches (the most common surface on tour).
"""
from __future__ import annotations

import re
from typing import Optional


# Tournament name keyword → surface. Keys are lowercase substrings; first
# match wins, so order matters (more specific first).
_TOUR_SURFACE_KEYWORDS: list[tuple[str, str]] = [
    # Clay
    ("roland garros", "Clay"),
    ("french open", "Clay"),
    ("monte carlo", "Clay"),
    ("monte-carlo", "Clay"),
    ("madrid", "Clay"),
    ("rome", "Clay"),
    ("italian open", "Clay"),
    ("internazionali", "Clay"),
    ("hamburg", "Clay"),
    ("estoril", "Clay"),
    ("barcelona", "Clay"),
    ("munich", "Clay"),
    ("bavarian", "Clay"),
    ("kitzbuhel", "Clay"),
    ("kitzbühel", "Clay"),
    ("umag", "Clay"),
    ("bastad", "Clay"),
    ("båstad", "Clay"),
    ("gstaad", "Clay"),
    ("buenos aires", "Clay"),
    ("rio open", "Clay"),
    ("santiago", "Clay"),
    ("cordoba", "Clay"),
    ("córdoba", "Clay"),
    ("houston", "Clay"),
    ("marrakech", "Clay"),
    ("geneva", "Clay"),
    ("lyon", "Clay"),
    ("clay", "Clay"),

    # Grass
    ("wimbledon", "Grass"),
    ("queen", "Grass"),
    ("halle", "Grass"),
    ("eastbourne", "Grass"),
    ("'s-hertogenbosch", "Grass"),
    ("hertogenbosch", "Grass"),
    ("rosmalen", "Grass"),
    ("stuttgart", "Grass"),
    ("mallorca", "Grass"),
    ("nottingham", "Grass"),
    ("birmingham", "Grass"),
    ("newport", "Grass"),
    ("grass", "Grass"),

    # Indoor / carpet (small set, mostly autumn European)
    ("paris masters", "Hard"),  # indoor hard
    ("rolex paris", "Hard"),
    ("bercy", "Hard"),
    ("vienna", "Hard"),
    ("erste bank", "Hard"),
    ("rotterdam", "Hard"),
    ("marseille", "Hard"),
    ("metz", "Hard"),
    ("antwerp", "Hard"),
    ("sofia", "Hard"),
    ("astana", "Hard"),

    # Hard outdoor (anchors)
    ("us open", "Hard"),
    ("australian open", "Hard"),
    ("indian wells", "Hard"),
    ("miami open", "Hard"),
    ("cincinnati", "Hard"),
    ("canadian open", "Hard"),
    ("rogers cup", "Hard"),
    ("national bank", "Hard"),
    ("national bank open", "Hard"),
    ("dubai", "Hard"),
    ("doha", "Hard"),
    ("acapulco", "Hard"),
    ("delray", "Hard"),
    ("auckland", "Hard"),
    ("brisbane", "Hard"),
    ("adelaide", "Hard"),
    ("united cup", "Hard"),
    ("atp finals", "Hard"),
    ("nitto atp finals", "Hard"),
    ("tour finals", "Hard"),
    ("hard", "Hard"),
]


def guess_surface_from_tournament(tournament: Optional[str]) -> str:
    """Return Hard / Clay / Grass best-effort from a tournament name."""
    if not tournament:
        return "Hard"
    t = tournament.lower()
    t = re.sub(r"\s+", " ", t)
    for needle, surface in _TOUR_SURFACE_KEYWORDS:
        if needle in t:
            return surface
    return "Hard"
