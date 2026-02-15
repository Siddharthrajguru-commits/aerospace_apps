@echo off
title Hydrogen-Electric UAV Propulsion Simulator
color 0A

cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo.
echo ============================================================
echo   Hydrogen-Electric UAV Propulsion Simulator Dashboard
echo ============================================================
echo.

REM Stop any existing Streamlit processes
echo [1/4] Stopping existing Streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

REM Check if app.py exists
if not exist "app.py" (
    echo ERROR: app.py not found in current directory!
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Start Streamlit server
echo [2/4] Starting Streamlit server on port 8502...
start /B streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false --server.runOnSave false

REM Wait for server to start
echo [3/4] Waiting for server to initialize...
timeout /t 6 /nobreak >nul

REM Open Chrome
echo [4/4] Opening Chrome browser...
set CHROME_FOUND=0

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8502
    set CHROME_FOUND=1
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:8502
    set CHROME_FOUND=1
) else if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" http://localhost:8502
    set CHROME_FOUND=1
)

if %CHROME_FOUND%==0 (
    echo WARNING: Chrome not found in standard locations.
    echo Opening default browser instead...
    start http://localhost:8502
)

echo.
echo ============================================================
echo   Dashboard is running!
echo   URL: http://localhost:8502
echo ============================================================
echo.
echo Keep this window open while using the dashboard.
echo Press Ctrl+C to stop the server.
echo.

REM Keep window open and wait
pause
