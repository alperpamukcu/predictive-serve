# src/utils/config.py

from pathlib import Path

# ---------------------------------------------------------
# PROJE DİZİN YAPISI
# ---------------------------------------------------------

# Bu dosya: PROJECT_ROOT/src/utils/config.py
# parents[0] = .../src/utils
# parents[1] = .../src
# parents[2] = .../predictive-serve   --> PROJE KÖKÜ
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# allyears.csv için canonical path
ALLYEARS_DIR: Path = RAW_DIR / "allyears"
ALLYEARS_PATH: Path = ALLYEARS_DIR / "allyears.csv"

# Varsayılan log dosyası (istersen kullanırsın)
DEFAULT_LOG_FILE: Path = LOGS_DIR / "app.log"
