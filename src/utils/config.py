# src/utils/config.py

from pathlib import Path
from dotenv import load_dotenv
import os

# Proje kök dizini (repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]

# .env dosyasını yükle
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ana data klasörü (varsayılan: data/)
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")

# Ham ve işlenmiş data klasörleri
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Tennis-data allyears dosyasının hedef yolu
ALLYEARS_PATH = RAW_DIR / "allyears" / "allyears.csv"

# (Eski) Sportsdata key - şu an kullanılmıyor ama dursa da olur
SPORTSDATA_API_KEY = os.getenv("SPORTSDATA_API_KEY", "")
