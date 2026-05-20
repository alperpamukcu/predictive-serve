@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem ============================================================
rem  PREDICTIVE SERVE - ONE CLICK LAUNCHER (WINDOWS)
rem ============================================================
echo.
echo ==========================================
echo   Predictive Serve - Full Pipeline + UI
echo ==========================================
echo.

rem Change working directory to the project root (folder of this script)
cd /d "%~dp0"

rem ------------------------------------------------------------
rem 1) Check Python (py launcher)
rem ------------------------------------------------------------
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python launcher "py" not found.
    echo         Please install Python from https://www.python.org/downloads/
    echo         and make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

echo [OK] Python launcher found.
for /f "tokens=*" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %PYTHON_VERSION%
echo.

rem ------------------------------------------------------------
rem 2) Install / update dependencies
rem ------------------------------------------------------------
echo [1/5] Installing Python dependencies from requirements.txt ...
py -m pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt
    echo         You can try manually:
    echo             py -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Dependencies are installed.
echo.

rem ------------------------------------------------------------
rem 3) Run full data and feature pipeline
rem     - Fetch 2000–2026 data from tennis-data.co.uk
rem     - Build processed datasets and train the model
rem ------------------------------------------------------------
echo ==========================================
echo   Running data and model pipeline
echo ==========================================
echo.

echo [2/5] Fetching raw match data (2000–2026) ...
rem (Pre-flight access check removed — fetch_data has its own retry/proxy logic.)
py -m src.data.fetch_data
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Data fetch failed. Please check your internet connection
    echo         or try running: py -m src.data.fetch_data
    pause
    exit /b 1
)
echo [OK] Raw data fetched.
echo.


echo [3/5] Preprocessing and cleaning ...
py -m src.data.preprocess
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Preprocess step failed.
    pause
    exit /b 1
)

py -m src.data.cleaning
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Cleaning step failed.
    pause
    exit /b 1
)
echo [OK] Preprocess and cleaning completed.
echo.

echo [4/5] Feature engineering (Elo, form, sets, market, etc.) ...
py -m src.features.elo
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Elo feature step failed.
    pause
    exit /b 1
)

py -m src.features.form
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Form feature step failed.
    pause
    exit /b 1
)

py -m src.features.build_features
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Building training dataset failed.
    pause
    exit /b 1
)
echo [OK] Feature datasets are ready.
echo.

echo [5/5] Training model and scoring all matches ...
py -m src.models.train_best
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)

py -m src.models.score_all_matches
if %errorlevel% neq 0 (
    echo.
    echo [WARN] Scoring all matches failed, UI will still start.
)

echo.
echo [INFO] Fetching upcoming fixtures (API-Tennis) ...
py -m src.data.fetch_upcoming_apitennis
if %errorlevel% neq 0 (
    echo [WARN] Upcoming fixtures fetch failed. Upcoming tab will show example or be empty.
)

echo.
echo [INFO] Enriching upcoming fixtures with odds (API-Tennis, optional) ...
py -m src.data.fetch_odds_apitennis
if %errorlevel% neq 0 (
    echo [WARN] Odds enrichment failed. Upcoming tab will still work without odds.
)

echo.
rem (Player image download is now part of the ATP roster sync below.)

echo.
echo [INFO] Syncing ATP roster + photos (API-Tennis) ...
py -m src.data.fetch_player_roster
py -m src.data.fetch_player_photos
echo [INFO] Player roster sync finished (best effort).

echo.
echo ==========================================
echo   Pipeline finished successfully
echo ==========================================
echo.
echo Starting Streamlit UI on http://localhost:8501 ...
echo (Press Ctrl+C in this window to stop the app.)
echo.

rem ------------------------------------------------------------
rem 4) Launch Streamlit UI
rem ------------------------------------------------------------
py -m streamlit run streamlit_app.py --server.headless true

echo.
echo Streamlit app has been closed.
pause

exit /b 0

