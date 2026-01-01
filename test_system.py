#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Serve - Sistem Test Scripti
Tum bilesenlerin calisip calismadigini kontrol eder.
"""

import sys
import os
from pathlib import Path
import traceback

# Windows'ta UTF-8 encoding için
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Renkli çıktı için
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL] {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.RESET}")

# Test sonuçları
tests_passed = 0
tests_failed = 0
tests_warnings = 0

def test_result(success: bool, msg: str, warning: bool = False):
    global tests_passed, tests_failed, tests_warnings
    if success:
        print_success(msg)
        tests_passed += 1
    elif warning:
        print_warning(msg)
        tests_warnings += 1
    else:
        print_error(msg)
        tests_failed += 1

# ============================================================================
# TEST 1: Proje Yapısı
# ============================================================================
print("\n" + "="*60)
print("TEST 1: Proje Yapısı Kontrolü")
print("="*60)

try:
    from src.utils.config import (
        PROJECT_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR,
        MODELS_DIR, ALLYEARS_PATH
    )
    test_result(True, "Config modülü yüklendi")
except Exception as e:
    test_result(False, f"Config modülü yüklenemedi: {e}")
    sys.exit(1)

# Klasör kontrolü
required_dirs = [
    (DATA_DIR, "data/"),
    (RAW_DIR, "data/raw/"),
    (PROCESSED_DIR, "data/processed/"),
    (MODELS_DIR, "models/"),
]
for dir_path, name in required_dirs:
    test_result(dir_path.exists(), f"{name} klasörü mevcut")

# ============================================================================
# TEST 2: Gerekli Dosyalar
# ============================================================================
print("\n" + "="*60)
print("TEST 2: Gerekli Dosyalar Kontrolü")
print("="*60)

required_files = [
    (PROCESSED_DIR / "train_dataset.csv", "train_dataset.csv"),
    (MODELS_DIR / "logreg_final.pkl", "logreg_final.pkl"),
    (MODELS_DIR / "imputer_final.pkl", "imputer_final.pkl"),
    (MODELS_DIR / "feature_columns.txt", "feature_columns.txt"),
]

for file_path, name in required_files:
    exists = file_path.exists()
    test_result(exists, f"{name} mevcut", warning=False)
    if exists:
        size = file_path.stat().st_size
        test_result(size > 0, f"{name} boş değil ({size:,} bytes)")

# Opsiyonel dosyalar
optional_files = [
    (PROCESSED_DIR / "all_predictions.csv", "all_predictions.csv"),
    (PROCESSED_DIR / "val_predictions.csv", "val_predictions.csv"),
    (PROCESSED_DIR / "matches_with_elo_form_sets.csv", "matches_with_elo_form_sets.csv"),
]

for file_path, name in optional_files:
    if file_path.exists():
        print_info(f"{name} mevcut (opsiyonel)")

# ============================================================================
# TEST 3: Python Modülleri Import
# ============================================================================
print("\n" + "="*60)
print("TEST 3: Python Modülleri Import Testi")
print("="*60)

modules_to_test = [
    ("src.data.fetch_data", "fetch_data"),
    ("src.data.preprocess", "preprocess"),
    ("src.data.cleaning", "cleaning"),
    ("src.features.elo", "elo"),
    ("src.features.form", "form"),
    ("src.features.build_features", "build_features"),
    ("src.models.train_logreg", "train_logreg"),
    ("src.models.score_all_matches", "score_all_matches"),
    ("src.predict.whatif", "whatif"),
    ("src.analysis.metrics", "metrics"),
]

for module_name, display_name in modules_to_test:
    try:
        __import__(module_name)
        test_result(True, f"{display_name} modülü import edildi")
    except Exception as e:
        test_result(False, f"{display_name} import hatası: {e}")

# ============================================================================
# TEST 4: Model Yükleme
# ============================================================================
print("\n" + "="*60)
print("TEST 4: Model Yükleme Testi")
print("="*60)

try:
    from joblib import load
    
    model_path = MODELS_DIR / "logreg_final.pkl"
    imputer_path = MODELS_DIR / "imputer_final.pkl"
    feature_cols_path = MODELS_DIR / "feature_columns.txt"
    
    if model_path.exists():
        try:
            model = load(model_path)
            test_result(True, "Model yüklendi")
            test_result(hasattr(model, 'predict_proba'), "Model predict_proba metoduna sahip")
        except Exception as e:
            test_result(False, f"Model yüklenemedi: {e}")
    
    if imputer_path.exists():
        try:
            imputer = load(imputer_path)
            test_result(True, "Imputer yüklendi")
        except Exception as e:
            test_result(False, f"Imputer yüklenemedi: {e}")
    
    if feature_cols_path.exists():
        try:
            with open(feature_cols_path, 'r', encoding='utf-8') as f:
                feature_cols = [line.strip() for line in f if line.strip()]
            test_result(len(feature_cols) > 0, f"Feature listesi okundu ({len(feature_cols)} feature)")
        except Exception as e:
            test_result(False, f"Feature listesi okunamadı: {e}")
            
except Exception as e:
    test_result(False, f"Model yükleme testi başarısız: {e}")

# ============================================================================
# TEST 5: Veri Seti Kontrolü
# ============================================================================
print("\n" + "="*60)
print("TEST 5: Veri Seti Kontrolü")
print("="*60)

try:
    import pandas as pd
    
    train_path = PROCESSED_DIR / "train_dataset.csv"
    if train_path.exists():
        df = pd.read_csv(train_path)
        test_result(len(df) > 0, f"train_dataset.csv okundu ({len(df):,} satır)")
        
        # Gerekli kolonlar
        required_cols = ["date", "playerA", "playerB", "y"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        test_result(len(missing_cols) == 0, 
                   f"Gerekli kolonlar mevcut: {', '.join(required_cols)}")
        
        if missing_cols:
            test_result(False, f"Eksik kolonlar: {', '.join(missing_cols)}")
        
        # Feature kolonları
        meta_cols = ["date", "surface", "playerA", "playerB", "y"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        test_result(len(feature_cols) > 0, f"Feature kolonları mevcut ({len(feature_cols)} adet)")
        
        # Tarih kontrolü
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            valid_dates = df["date"].notna().sum()
            test_result(valid_dates > 0, f"Geçerli tarihler: {valid_dates:,}/{len(df):,}")
            
except Exception as e:
    test_result(False, f"Veri seti kontrolü başarısız: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 6: Feature Engineering Fonksiyonları
# ============================================================================
print("\n" + "="*60)
print("TEST 6: Feature Engineering Fonksiyonları")
print("="*60)

try:
    from src.features.elo import compute_elo_for_matches, expected_score
    test_result(True, "Elo fonksiyonları import edildi")
    
    # Basit Elo testi
    score = expected_score(1500, 1500)
    test_result(abs(score - 0.5) < 0.01, f"Elo expected_score testi (1500 vs 1500 = {score:.3f})")
    
except Exception as e:
    test_result(False, f"Elo fonksiyonları testi başarısız: {e}")

# ============================================================================
# TEST 7: Tahmin Fonksiyonları
# ============================================================================
print("\n" + "="*60)
print("TEST 7: Tahmin Fonksiyonları")
print("="*60)

try:
    from src.predict.whatif import build_feature_row
    from src.utils.feature_utils import load_feature_list
    test_result(True, "What-if fonksiyonları import edildi")
    
    # Feature list yükleme
    feature_cols_path = MODELS_DIR / "feature_columns.txt"
    if feature_cols_path.exists():
        try:
            feature_cols = load_feature_list(feature_cols_path)
            test_result(len(feature_cols) > 0, f"Feature listesi yüklendi ({len(feature_cols)} feature)")
        except Exception as e:
            test_result(False, f"Feature listesi yüklenemedi: {e}")
            
except Exception as e:
    test_result(False, f"Tahmin fonksiyonları testi başarısız: {e}")

# ============================================================================
# TEST 8: Metrik Hesaplama
# ============================================================================
print("\n" + "="*60)
print("TEST 8: Metrik Hesaplama")
print("="*60)

try:
    from src.analysis.metrics import compute_overall_metrics
    import pandas as pd
    import numpy as np
    
    # Test verisi oluştur
    test_df = pd.DataFrame({
        'y': [1, 0, 1, 0, 1],
        'p_model': [0.7, 0.3, 0.8, 0.2, 0.6],
        'pA_market': [0.65, 0.35, 0.75, 0.25, 0.55]
    })
    
    metrics = compute_overall_metrics(test_df)
    test_result(True, "Metrik hesaplama çalışıyor")
    test_result(metrics.model_logloss > 0, f"Model logloss: {metrics.model_logloss:.4f}")
    test_result(metrics.market_logloss is not None, "Market logloss hesaplandı")
    
except Exception as e:
    test_result(False, f"Metrik hesaplama testi başarısız: {e}")

# ============================================================================
# TEST 9: Streamlit Uygulaması
# ============================================================================
print("\n" + "="*60)
print("TEST 9: Streamlit Uygulaması Kontrolü")
print("="*60)

streamlit_file = PROJECT_ROOT / "streamlit_app.py"
test_result(streamlit_file.exists(), "streamlit_app.py mevcut")

if streamlit_file.exists():
    try:
        # Streamlit import testi
        import streamlit as st
        test_result(True, "Streamlit modülü yüklendi")
    except ImportError:
        test_result(False, "Streamlit modülü yüklenemedi (pip install streamlit)")

# ============================================================================
# TEST 10: End-to-End Pipeline Kontrolü
# ============================================================================
print("\n" + "="*60)
print("TEST 10: End-to-End Pipeline Kontrolü")
print("="*60)

# Pipeline adımları kontrolü
pipeline_steps = [
    (RAW_DIR / "allyears.csv", "1. Ham veri (allyears.csv)"),
    (PROCESSED_DIR / "matches_allyears.csv", "2. Preprocess ciktisi"),
    (PROCESSED_DIR / "matches_clean.csv", "3. Cleaning ciktisi"),
    (PROCESSED_DIR / "matches_with_elo.csv", "4. Elo feature'lari"),
    (PROCESSED_DIR / "matches_with_elo_form.csv", "5. Form feature'lari"),
    (PROCESSED_DIR / "matches_with_elo_form_sets.csv", "6. Set feature'lari"),
    (PROCESSED_DIR / "train_dataset.csv", "7. Final feature seti"),
]

pipeline_ok = True
for file_path, step_name in pipeline_steps:
    exists = file_path.exists()
    if exists:
        size = file_path.stat().st_size
        print_info(f"{step_name}: OK ({size:,} bytes)")
    else:
        print_warning(f"{step_name}: EKSIK")
        pipeline_ok = False

test_result(pipeline_ok, "Pipeline dosyaları mevcut", warning=not pipeline_ok)

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*60)
print("TEST ÖZETİ")
print("="*60)

total_tests = tests_passed + tests_failed + tests_warnings
print(f"\nToplam Test: {total_tests}")
print_success(f"Başarılı: {tests_passed}")
if tests_warnings > 0:
    print_warning(f"Uyarı: {tests_warnings}")
if tests_failed > 0:
    print_error(f"Başarısız: {tests_failed}")

success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
print(f"\nBaşarı Oranı: {success_rate:.1f}%")

if tests_failed == 0:
    print_success("\nTum kritik testler basarili!")
    sys.exit(0)
else:
    print_error(f"\n{tests_failed} test basarisiz. Lutfen hatalari duzeltin.")
    sys.exit(1)

