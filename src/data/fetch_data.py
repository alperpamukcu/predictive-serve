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


BASE_URL = "http://www.tennis-data.co.uk"  # eski proje kodundaki URL
START_YEAR = 2000
END_YEAR = 2025  # istersen bunu güncelleyebilirsin


def download_season(year: int) -> pd.DataFrame:
    """
    Verilen yıl için tennis-data.co.uk'den .xlsx dosyasını indirir ve DataFrame'e çevirir.
    Örn: http://www.tennis-data.co.uk/2020/2020.xlsx
    """
    url = f"{BASE_URL}/{year}/{year}.xlsx"
    print(f"[fetch_data] Downloading {year} from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Excel içeriğini hafızaya alıp pandas ile oku
    buffer = BytesIO(resp.content)
    df = pd.read_excel(buffer)
    df["source_year"] = year
    return df


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

    all_seasons: List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        try:
            df_year = download_season(year)
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
