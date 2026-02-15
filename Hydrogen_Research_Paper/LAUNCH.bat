@echo off
title Hydrogen Dashboard Server
color 0A

cd /d "c:\GEN AI Projects\Hydrogen_Research_Paper"

echo.
echo ========================================
echo   Hydrogen-Electric UAV Dashboard
echo ========================================
echo.

REM Stop existing processes
taskkill /F /IM streamlit.exe 2>nul
timeout /t 1 /nobreak >nul

echo Starting server on http://localhost:8502
echo.
echo This window must stay open for the dashboard to work.
echo Close this window to stop the server.
echo.

REM Run Streamlit - this will keep the window open
python -m streamlit run app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false

REM After Streamlit closes
echo.
echo Server stopped.
pause
