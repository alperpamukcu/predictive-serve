from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Proje kök dizini: .../predictive-serve
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ana klasörler
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ALLYEARS_DIR = RAW_DIR / "allyears"
ALLYEARS_PATH = ALLYEARS_DIR / "allyears.csv"

# Gerekli klasörleri oluştur
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# .env dosyasını yükle (varsa)
env_path = PROJECT_ROOT / ".env"
if load_dotenv is not None and env_path.exists():
    load_dotenv(env_path)
