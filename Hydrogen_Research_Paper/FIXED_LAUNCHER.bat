@echo off
title Hydrogen-Electric UAV Dashboard Launcher
color 0B

cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo.
echo ============================================================
echo   Hydrogen-Electric UAV Propulsion Simulator
echo   Dashboard Launcher
echo ============================================================
echo.

REM Kill all Streamlit and Python processes related to this
echo [Step 1/5] Stopping all existing Streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
taskkill /F /FI "WINDOWTITLE eq streamlit*" 2>nul
timeout /t 2 /nobreak >nul

REM Check if app.py exists
echo [Step 2/5] Checking files...
if not exist "app.py" (
    echo ERROR: app.py not found!
    echo Current directory: %CD%
    pause
    exit /b 1
)
echo    ✓ app.py found

REM Check if core directory exists
if not exist "core\" (
    echo ERROR: core directory not found!
    pause
    exit /b 1
)
echo    ✓ core directory found

REM Test Python imports
echo [Step 3/5] Testing Python imports...
python -c "import streamlit; import numpy; import matplotlib; print('    ✓ All imports OK')" 2>nul
if errorlevel 1 (
    echo ERROR: Python imports failed!
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Start Streamlit with explicit Python path
echo [Step 4/5] Starting Streamlit server on port 8502...
echo    This may take 10-15 seconds...
echo.

REM Use full path to streamlit
python -m streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false --server.runOnSave false --server.address localhost

REM The above command will block, so we need a different approach
start "Streamlit Server" cmd /k "python -m streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false --server.runOnSave false --server.address localhost"

REM Wait for server to start
echo [Step 5/5] Waiting for server to start (15 seconds)...
timeout /t 15 /nobreak >nul

REM Check if port is listening
netstat -ano | findstr ":8502" >nul
if errorlevel 1 (
    echo.
    echo WARNING: Port 8502 may not be ready yet.
    echo Please wait a few more seconds and try opening:
    echo http://localhost:8502
    echo.
) else (
    echo    ✓ Server is running on port 8502
)

REM Open Chrome
echo.
echo Opening Chrome browser...
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else (
    echo Chrome not found. Opening default browser...
    start http://localhost:8502
)

echo.
echo ============================================================
echo   Dashboard URL: http://localhost:8502
echo ============================================================
echo.
echo Keep the "Streamlit Server" window open while using the dashboard.
echo Close it to stop the server.
echo.
pause
