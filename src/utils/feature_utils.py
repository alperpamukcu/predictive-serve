# src/utils/feature_utils.py

from pathlib import Path
from typing import List


def load_feature_list(path: Path) -> List[str]:
    """
    Feature kolon listesini dosyadan okur.
    
    Args:
        path: feature_columns.txt dosyasının yolu
        
    Returns:
        Feature kolon isimlerinin listesi
    """
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]



