@echo off
cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo.
echo Starting Hydrogen-Electric UAV Dashboard...
echo.

REM Kill existing processes
taskkill /F /IM streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

REM Start Streamlit in visible window
start "Streamlit Dashboard Server" cmd /k "python -m streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false"

REM Wait
timeout /t 10 /nobreak >nul

REM Open Chrome
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:8502
) else (
    start http://localhost:8502
)

echo.
echo Dashboard should be opening in Chrome...
echo If it doesn't open, go to: http://localhost:8502
echo.
echo Keep the "Streamlit Dashboard Server" window open.
echo.
pause
