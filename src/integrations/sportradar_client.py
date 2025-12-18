# src/integrations/sportradar_client.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import requests
import pandas as pd

from src.utils.config import SPORTRADAR_API_KEY

SPORTRADAR_BASE_URL = "https://api.sportradar.com/tennis/trial/v3/en"  # ÖRNEK, dokümana göre değişebilir


@dataclass
class UpcomingMatch:
    event_id: str
    date_utc: datetime
    tourney_name: str
    round_name: str
    playerA_name: str
    playerB_name: str
    surface: str | None = None


def fetch_upcoming_matches(days_ahead: int = 3) -> pd.DataFrame:
    """
    Sportradar'dan önümüzdeki X gün içindeki maçları çeker ve
    internal bir DataFrame formatına çevirir.

    Şimdilik sadece iskelet: endpoint ve response yapısı,
    dokümana göre doldurulacak.
    """
    if not SPORTRADAR_API_KEY:
        raise RuntimeError("SPORTRADAR_API_KEY .env içinde tanımlı değil")

    # TODO:
    # 1) Dokümana göre doğru endpoint'i belirle
    # 2) requests.get ile çağrı yap
    # 3) JSON'u parse edip UpcomingMatch listesine çevir
    # 4) Oradan da pd.DataFrame'e dönüştür

    # Şimdilik boş DataFrame döndürüyoruz (MVP iskeleti)
    cols = [
        "event_id",
        "date_utc",
        "tourney_name",
        "round_name",
        "playerA_name",
        "playerB_name",
        "surface",
    ]
    return pd.DataFrame(columns=cols)
