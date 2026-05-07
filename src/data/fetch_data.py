# src/data/fetch_data.py

"""
Tennis-data.co.uk'den 2000-2025 arası sezonları indirip
tek bir allyears.csv dosyasında birleştiren script.

Elle dosya indirmeye gerek yok; bu script:
- Her yılın .xlsx dosyasını indirir
- Hepsini tek bir DataFrame'de birleştirir
- data/raw/allyears.csv olarak kaydeder
"""

from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.utils.config import RAW_DIR, ALLYEARS_PATH
from src.utils.config import PROJECT_ROOT
from src.utils.env import try_load_dotenv, getenv, getenv_int


BASE_URL = "http://www.tennis-data.co.uk"  # default
START_YEAR = 2000
END_YEAR = 2026  # güncel sezon dahil


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            # be explicit; some proxies/hosts behave differently without UA
            "User-Agent": "predictive-serve/1.0 (+https://example.invalid)",
            "Accept": "*/*",
        }
    )
    return s


def _proxies_from_env() -> dict | None:
    """
    Support a single proxy env var for this datasource.
    If TEN_DATA_PROXY is set, use it for both http/https.
    """
    p = getenv("TEN_DATA_PROXY")
    if not p:
        return None
    return {"http": p, "https": p}


def download_season(year: int, sess: requests.Session, base_url: str, timeout_s: int, retries: int) -> pd.DataFrame:
    """
    Download a season's archive from tennis-data.co.uk.

    The site keeps recent years in ``.xlsx`` and pre-2013 years in legacy
    ``.xls``. We try the modern format first and fall back to ``.xls`` (which
    requires the ``xlrd`` package).
    """
    proxies = _proxies_from_env()
    last_err: Exception | None = None
    for ext, engine in [("xlsx", "openpyxl"), ("xls", "xlrd")]:
        url = f"{base_url.rstrip('/')}/{year}/{year}.{ext}"
        print(f"[fetch_data] Downloading {year} from {url}")
        for attempt in range(1, max(1, retries) + 1):
            try:
                resp = sess.get(url, timeout=timeout_s, proxies=proxies, allow_redirects=True)
                resp.raise_for_status()
                # If the redirect lands on a different extension, the engine still applies.
                buffer = BytesIO(resp.content)
                try:
                    df = pd.read_excel(buffer, engine=engine)
                except ImportError:
                    # xlrd missing — fall back to default engine inference
                    buffer.seek(0)
                    df = pd.read_excel(buffer)
                df["source_year"] = year
                return df
            except Exception as e:
                last_err = e
                print(f"[fetch_data] WARNING: {year}.{ext} attempt {attempt}/{retries} failed: {e}")
    assert last_err is not None
    raise last_err


def build_allyears_csv(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    output_path: Path = ALLYEARS_PATH,
) -> Path:
    """
    Belirtilen yıl aralığındaki tüm sezonları indirir,
    tek DataFrame'de birleştirir ve allyears.csv olarak kaydeder.
    """
    # Klasörü oluştur
    allyears_dir = output_path.parent
    allyears_dir.mkdir(parents=True, exist_ok=True)

    try_load_dotenv(PROJECT_ROOT)
    base_url = getenv("TEN_DATA_BASE_URL", BASE_URL) or BASE_URL
    timeout_s = int(getenv_int("TEN_DATA_TIMEOUT_S", 30))
    retries = int(getenv_int("TEN_DATA_RETRIES", 3))

    sess = _session()
    all_seasons: List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        try:
            df_year = download_season(year, sess=sess, base_url=base_url, timeout_s=timeout_s, retries=retries)
            all_seasons.append(df_year)
        except Exception as e:
            print(f"[fetch_data] WARNING: {year} indirilemedi: {e}")

    if not all_seasons:
        raise RuntimeError("Hiçbir sezon indirilemedi, allyears.csv oluşturulamadı.")

    raw_matches = pd.concat(all_seasons, ignore_index=True)

    # Tarih kolonunu normalize et (bazı yıllarda Date sütunu string olabilir)
    if "Date" in raw_matches.columns:
        raw_matches["Date"] = pd.to_datetime(raw_matches["Date"], errors="coerce").dt.date

    # Çıktıyı kaydet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_matches.to_csv(output_path, index=False)
    print(f"[fetch_data] Saved merged allyears to: {output_path} (rows={len(raw_matches)})")

    return output_path


if __name__ == "__main__":
    build_allyears_csv()
