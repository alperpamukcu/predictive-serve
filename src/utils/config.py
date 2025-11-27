from pathlib import Path
from dotenv import load_dotenv
import os

# Proje kök dizini (repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]

# .env dosyasını yükle
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")

# Sportsdata.io API key
SPORTSDATA_API_KEY = os.getenv("SPORTSDATA_API_KEY", "")
