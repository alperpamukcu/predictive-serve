# -*- coding: utf-8 -*-
"""
Hizli Test - Sadece kritik kontroller
"""

import sys
from pathlib import Path

def test_basic():
    """Temel dosya ve klasor kontrolleri"""
    print("=" * 60)
    print("HIZLI TEST - Temel Kontroller")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Config kontrolu
    try:
        from src.utils.config import PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, MODELS_DIR
        print("[OK] Config modulu yuklendi")
    except Exception as e:
        errors.append(f"Config modulu: {e}")
        print(f"[FAIL] Config modulu: {e}")
        return
    
    # Klasorler
    dirs = [
        (DATA_DIR, "data/"),
        (PROCESSED_DIR, "data/processed/"),
        (MODELS_DIR, "models/"),
    ]
    for dir_path, name in dirs:
        if dir_path.exists():
            print(f"[OK] {name} klasoru mevcut")
        else:
            errors.append(f"{name} klasoru eksik")
            print(f"[FAIL] {name} klasoru eksik")
    
    # Kritik dosyalar
    critical_files = [
        (PROCESSED_DIR / "train_dataset.csv", "train_dataset.csv"),
        (MODELS_DIR / "logreg_final.pkl", "logreg_final.pkl"),
        (MODELS_DIR / "imputer_final.pkl", "imputer_final.pkl"),
        (MODELS_DIR / "feature_columns.txt", "feature_columns.txt"),
    ]
    
    for file_path, name in critical_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"[OK] {name} mevcut ({size:,} bytes)")
        else:
            errors.append(f"{name} eksik")
            print(f"[FAIL] {name} eksik")
    
    # Opsiyonel dosyalar
    optional_files = [
        (PROCESSED_DIR / "all_predictions.csv", "all_predictions.csv"),
        (PROCESSED_DIR / "val_predictions.csv", "val_predictions.csv"),
    ]
    
    for file_path, name in optional_files:
        if file_path.exists():
            print(f"[INFO] {name} mevcut (opsiyonel)")
        else:
            warnings.append(f"{name} eksik (opsiyonel)")
    
    # Ozet
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    
    if not errors:
        print("[OK] Tum kritik dosyalar mevcut!")
        if warnings:
            print(f"[WARN] {len(warnings)} opsiyonel dosya eksik")
        return True
    else:
        print(f"[FAIL] {len(errors)} kritik hata bulundu:")
        for err in errors:
            print(f"  - {err}")
        return False

if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)



