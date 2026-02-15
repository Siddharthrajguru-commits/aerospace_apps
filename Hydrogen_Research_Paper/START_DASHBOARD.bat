@echo off
cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

REM Stop any existing Streamlit
taskkill /F /IM streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

REM Start Streamlit
start /B streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false

REM Wait for server
timeout /t 6 /nobreak >nul

REM Open Chrome
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else (
    start http://localhost:8502
)

echo Dashboard started! Check Chrome browser.
pause
