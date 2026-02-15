# PowerShell script to properly launch Streamlit dashboard
Set-Location "c:\GEN AI Projects\Hydrogen_Research_Paper"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Hydrogen-Electric UAV Propulsion Simulator Dashboard" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop all existing Streamlit processes
Write-Host "[1/6] Stopping existing Streamlit processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "    ✓ All processes stopped" -ForegroundColor Green

# Step 2: Check files
Write-Host "[2/6] Checking required files..." -ForegroundColor Yellow
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found!" -ForegroundColor Red
    exit 1
}
Write-Host "    ✓ app.py found" -ForegroundColor Green

if (-not (Test-Path "core\")) {
    Write-Host "ERROR: core directory not found!" -ForegroundColor Red
    exit 1
}
Write-Host "    ✓ core directory found" -ForegroundColor Green

# Step 3: Test imports
Write-Host "[3/6] Testing Python imports..." -ForegroundColor Yellow
try {
    python -c "import streamlit; import numpy; import matplotlib; from core.fuel_cell import PEMFuelCell; from core.tank import LH2Tank; from core.mission import MissionProfile; print('OK')" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Import failed"
    }
    Write-Host "    ✓ All imports successful" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python imports failed!" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Step 4: Start Streamlit in a new window
Write-Host "[4/6] Starting Streamlit server..." -ForegroundColor Yellow
Write-Host "    Port: 8502" -ForegroundColor Cyan
Write-Host "    This will open in a new window..." -ForegroundColor Cyan

$streamlitArgs = @(
    "run",
    "app.py",
    "--server.port", "8502",
    "--server.headless", "false",
    "--browser.gatherUsageStats", "false",
    "--server.runOnSave", "false",
    "--server.address", "localhost"
)

# Start Streamlit in a new window so we can see errors
$process = Start-Process -FilePath "python" -ArgumentList "-m", "streamlit" + $streamlitArgs -PassThru -WindowStyle Normal

if (-not $process) {
    Write-Host "ERROR: Failed to start Streamlit!" -ForegroundColor Red
    exit 1
}

Write-Host "    ✓ Streamlit process started (PID: $($process.Id))" -ForegroundColor Green

# Step 5: Wait for server to be ready
Write-Host "[5/6] Waiting for server to be ready..." -ForegroundColor Yellow
$maxWait = 20
$waited = 0
$serverReady = $false

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 1
    $waited++
    
    # Check if port 8502 is listening
    $connection = Test-NetConnection -ComputerName localhost -Port 8502 -WarningAction SilentlyContinue -InformationLevel Quiet
    
    if ($connection) {
        $serverReady = $true
        Write-Host "    ✓ Server is ready!" -ForegroundColor Green
        break
    }
    
    Write-Host "." -NoNewline -ForegroundColor Gray
}

Write-Host ""

if (-not $serverReady) {
    Write-Host "WARNING: Server may not be ready yet. Please wait a few more seconds." -ForegroundColor Yellow
    Write-Host "Check the Streamlit window for any error messages." -ForegroundColor Yellow
}

# Step 6: Open Chrome
Write-Host "[6/6] Opening Chrome browser..." -ForegroundColor Yellow

$chromePaths = @(
    "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe"
)

$chromeOpened = $false
foreach ($path in $chromePaths) {
    if (Test-Path $path) {
        Start-Process -FilePath $path -ArgumentList "http://localhost:8502"
        Write-Host "    ✓ Chrome opened!" -ForegroundColor Green
        $chromeOpened = $true
        break
    }
}

if (-not $chromeOpened) {
    Write-Host "    Chrome not found. Opening default browser..." -ForegroundColor Yellow
    Start-Process "http://localhost:8502"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Dashboard URL: http://localhost:8502" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Keep the Streamlit window open while using the dashboard." -ForegroundColor Yellow
Write-Host "Close it to stop the server." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit this launcher (server will keep running)..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
