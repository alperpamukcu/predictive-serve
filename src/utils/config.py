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
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# allyears.csv için canonical path
ALLYEARS_PATH: Path = RAW_DIR / "allyears.csv"
