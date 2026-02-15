@echo off
cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo ========================================
echo Hydrogen-Electric UAV Propulsion Simulator
echo ========================================
echo.

echo Stopping any existing Streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

echo Starting Streamlit server on port 8502...
echo.

start /B streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false

echo Waiting for server to start...
timeout /t 5 /nobreak >nul

echo Opening Chrome browser...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8502
if errorlevel 1 (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:8502
    if errorlevel 1 (
        start http://localhost:8502
    )
)

echo.
echo Dashboard URL: http://localhost:8502
echo.
echo Press Ctrl+C to stop the server
echo.

pause
