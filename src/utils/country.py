"""
Map API-Tennis ``player_country`` strings to ISO-3166 alpha-2 country codes
and the corresponding flag emoji.
"""
from __future__ import annotations

import re
import unicodedata


# Curated mapping for the country names API-Tennis returns. Anything missing
# falls through to a ?? code and a generic globe icon.
_NAME_TO_CODE: dict[str, str] = {
    "argentina": "AR",
    "australia": "AU",
    "austria": "AT",
    "belarus": "BY",
    "belgium": "BE",
    "bolivia": "BO",
    "bosnia and herzegovina": "BA",
    "brazil": "BR",
    "bulgaria": "BG",
    "canada": "CA",
    "chile": "CL",
    "china": "CN",
    "colombia": "CO",
    "croatia": "HR",
    "cuba": "CU",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "ecuador": "EC",
    "egypt": "EG",
    "estonia": "EE",
    "finland": "FI",
    "france": "FR",
    "georgia": "GE",
    "germany": "DE",
    "great britain": "GB",
    "united kingdom": "GB",
    "uk": "GB",
    "greece": "GR",
    "hungary": "HU",
    "india": "IN",
    "indonesia": "ID",
    "iran": "IR",
    "ireland": "IE",
    "israel": "IL",
    "italy": "IT",
    "japan": "JP",
    "kazakhstan": "KZ",
    "korea": "KR",
    "south korea": "KR",
    "latvia": "LV",
    "lithuania": "LT",
    "luxembourg": "LU",
    "mexico": "MX",
    "moldova": "MD",
    "monaco": "MC",
    "morocco": "MA",
    "netherlands": "NL",
    "new zealand": "NZ",
    "norway": "NO",
    "paraguay": "PY",
    "peru": "PE",
    "philippines": "PH",
    "poland": "PL",
    "portugal": "PT",
    "qatar": "QA",
    "romania": "RO",
    "russia": "RU",
    "russian federation": "RU",
    "saudi arabia": "SA",
    "serbia": "RS",
    "slovakia": "SK",
    "slovenia": "SI",
    "south africa": "ZA",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "taiwan": "TW",
    "chinese taipei": "TW",
    "thailand": "TH",
    "tunisia": "TN",
    "turkey": "TR",
    "türkiye": "TR",
    "ukraine": "UA",
    "united arab emirates": "AE",
    "united states": "US",
    "usa": "US",
    "uruguay": "UY",
    "uzbekistan": "UZ",
    "venezuela": "VE",
    "vietnam": "VN",
    "zimbabwe": "ZW",
}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def country_code(country_name: str | None) -> str | None:
    if not country_name:
        return None
    norm = _normalize(country_name)
    if norm in _NAME_TO_CODE:
        return _NAME_TO_CODE[norm]
    # already an ISO-2?
    if re.fullmatch(r"[a-z]{2}", norm):
        return norm.upper()
    return None


def flag_emoji(country_name: str | None) -> str:
    code = country_code(country_name)
    if not code:
        return "🏳️"
    # Regional indicator letters: 'A' (0x41) -> 0x1F1E6
    return "".join(chr(0x1F1E6 + (ord(c) - ord("A"))) for c in code)
