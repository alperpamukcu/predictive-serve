@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   PREDICTIVE SERVE - FULL PIPELINE
echo ========================================
echo.
echo Bu script:
echo   - Tennis-data.co.uk'den güncel verileri çeker
echo   - Tüm veri işleme adımlarını çalıştırır
echo   - Modeli eğitir ve tahminler yapar
echo   - Streamlit web arayüzünü başlatır
echo.
echo ========================================
echo.

REM Python kontrolü (py launcher kullanılıyor)
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadı! Lütfen Python'u yükleyin.
    echo       Python'u https://www.python.org/downloads/ adresinden indirebilirsiniz.
    echo       Kurulum sırasında "Add Python to PATH" seçeneğini işaretlemeyi unutmayın!
    pause
    exit /b 1
)

echo [OK] Python bulundu
for /f "tokens=*" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %PYTHON_VERSION%
echo.

REM Gerekli paketlerin kontrolü
echo [1/11] Gerekli paketler kontrol ediliyor...
py -c "import pandas, numpy, sklearn, streamlit, joblib, requests, openpyxl" >nul 2>&1
if %errorlevel% neq 0 (
    echo [UYARI] Bazı paketler eksik. Yükleniyor...
    echo        Bu işlem birkaç dakika sürebilir...
    py -m pip install -r requirements.txt --quiet --disable-pip-version-check
    if !errorlevel! neq 0 (
        echo [HATA] Paket yükleme başarısız!
        echo       Lütfen manuel olarak çalıştırın: py -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Paketler yüklendi
) else (
    echo [OK] Tüm paketler yüklü
)

REM Proje dizinine geç
cd /d "%~dp0"

echo.
echo ========================================
echo   VERİ İŞLEME AŞAMASI
echo ========================================
echo.

echo [2/11] Veri toplama başlatılıyor (tennis-data.co.uk)...
echo        Tennis-data.co.uk'den 2000-2025 yılları arası veriler indiriliyor...
echo        Bu işlem birkaç dakika sürebilir (internet hızına bağlı)...
py -m src.data.fetch_data
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Veri toplama başarısız!
    echo       Lütfen internet bağlantınızı kontrol edin.
    pause
    exit /b 1
)
echo [OK] Veri toplama tamamlandı
echo.

echo [3/11] Veri ön işleme (preprocess)...
echo        Ham veri normalize ediliyor...
py -m src.data.preprocess
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Preprocess başarısız!
    pause
    exit /b 1
)
echo [OK] Preprocess tamamlandı
echo.

echo [4/11] Veri temizleme (cleaning)...
echo        Eksik ve hatalı veriler temizleniyor...
py -m src.data.cleaning
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Cleaning başarısız!
    pause
    exit /b 1
)
echo [OK] Cleaning tamamlandı
echo.

echo ========================================
echo   FEATURE ENGINEERING AŞAMASI
echo ========================================
echo.

echo [5/11] Elo rating hesaplama...
echo        Oyuncuların güç skorları hesaplanıyor...
py -m src.features.elo
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Elo hesaplama başarısız!
    pause
    exit /b 1
)
echo [OK] Elo rating hesaplama tamamlandı
echo.

echo [6/11] Form feature'ları hesaplama...
echo        Oyuncuların son performansları analiz ediliyor...
py -m src.features.form
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Form feature hesaplama başarısız!
    pause
    exit /b 1
)
echo [OK] Form feature'ları hesaplama tamamlandı
echo.

echo [7/11] Set feature'ları hesaplama...
echo        Set bazlı performans istatistikleri hesaplanıyor...
py -m src.features.sets
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Set feature hesaplama başarısız!
    pause
    exit /b 1
)
echo [OK] Set feature'ları hesaplama tamamlandı
echo.

echo [8/11] Tüm feature'ları birleştirme (build_features)...
echo        H2H, Market, Round ve diğer feature'lar ekleniyor...
py -m src.features.build_features
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Feature birleştirme başarısız!
    pause
    exit /b 1
)
echo [OK] Feature birleştirme tamamlandı
echo.

echo ========================================
echo   MODEL EĞİTİMİ AŞAMASI
echo ========================================
echo.

echo [9/11] Model eğitimi (Logistic Regression)...
echo        Model eğitiliyor ve performans metrikleri hesaplanıyor...
echo        Bu işlem birkaç dakika sürebilir...
py -m src.models.train_logreg
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Model eğitimi başarısız!
    pause
    exit /b 1
)
echo [OK] Model eğitimi tamamlandı
echo.

echo [10/11] Tüm maçlara tahmin yapılıyor...
echo         Geçmiş tüm maçlar için model tahminleri oluşturuluyor...
py -m src.models.score_all_matches
if %errorlevel% neq 0 (
    echo [UYARI] Tahmin yapma başarısız, ancak devam ediliyor...
    echo         Streamlit uygulaması yine de çalışacaktır.
) else (
    echo [OK] Tahmin işlemi tamamlandı
)
echo.

echo.
echo ========================================
echo   PIPELINE TAMAMLANDI!
echo ========================================
echo.
echo [11/11] Streamlit web arayüzü başlatılıyor...
echo.
echo ========================================
echo   UYGULAMA HAZIR!
echo ========================================
echo.
echo Tarayıcınızda otomatik olarak açılacaktır.
echo Eğer açılmazsa, tarayıcınızda şu adresi açın:
echo    http://localhost:8501
echo.
echo Uygulamayı kapatmak için bu pencerede Ctrl+C tuşlarına basın.
echo.
echo ========================================
echo.

REM Streamlit uygulamasını başlat
py -m streamlit run streamlit_app.py

REM Eğer streamlit kapanırsa
echo.
echo Streamlit uygulaması kapatıldı.
pause

