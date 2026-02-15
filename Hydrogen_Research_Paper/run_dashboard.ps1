# PowerShell script to run the Streamlit dashboard
Set-Location "c:\GEN AI Projects\Hydrogen_Research_Paper"

Write-Host "Starting Hydrogen-Electric UAV Propulsion Simulator Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "The dashboard will open in your browser at http://localhost:8502" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Stop any existing Streamlit processes
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Run Streamlit on port 8502
streamlit run app.py --server.port 8502 --server.headless false
