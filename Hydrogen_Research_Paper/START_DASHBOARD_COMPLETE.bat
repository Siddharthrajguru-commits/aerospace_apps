@echo off
title Dashboard Launcher
color 0B

cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo.
echo ============================================================
echo   Hydrogen-Electric UAV Propulsion Simulator
echo   Complete Launcher
echo ============================================================
echo.

REM Step 1: Stop existing
echo [1/3] Stopping existing servers...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul
echo    Done!
echo.

REM Step 2: Start server in new window
echo [2/3] Starting Streamlit server...
echo    This will open in a separate window.
echo    Please wait 10 seconds for the server to start...
echo.
start "Hydrogen Dashboard Server" cmd /k "cd /d c:\GEN AI Projects\Hydrogen_Research_Paper && python -m streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false"

REM Step 3: Wait and open Chrome
echo [3/3] Waiting for server to start...
timeout /t 10 /nobreak >nul

echo    Opening Chrome browser...
call OPEN_CHROME.bat

echo.
echo ============================================================
echo   Dashboard should now be open in Chrome!
echo   URL: http://localhost:8502
echo ============================================================
echo.
echo IMPORTANT: Keep the "Hydrogen Dashboard Server" window open!
echo Close it only when you're done using the dashboard.
echo.
pause
