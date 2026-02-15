# PowerShell script to run Streamlit dashboard and open in Chrome
Set-Location "c:\GEN AI Projects\Hydrogen_Research_Paper"

Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host " Hydrogen-Electric UAV Propulsion Simulator " -NoNewline -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host "=" -ForegroundColor Green
Write-Host ""

# Stop any existing Streamlit processes
Write-Host "Stopping any existing Streamlit processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting Streamlit server on port 8502..." -ForegroundColor Green
Write-Host ""

# Set the URL
$url = "http://localhost:8502"

# Start Streamlit in background
$streamlitProcess = Start-Process -FilePath "streamlit" -ArgumentList "run", "app.py", "--server.port", "8502", "--server.headless", "false", "--browser.gatherUsageStats", "false" -PassThru -NoNewWindow

# Wait a few seconds for Streamlit to start
Write-Host "Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if Streamlit is running
if ($streamlitProcess.HasExited) {
    Write-Host "ERROR: Streamlit failed to start!" -ForegroundColor Red
    exit 1
}

# Open Chrome with the URL
Write-Host "Opening Chrome browser..." -ForegroundColor Green
try {
    # Try to find Chrome in common locations
    $chromePaths = @(
        "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe"
    )
    
    $chromeFound = $false
    foreach ($path in $chromePaths) {
        if (Test-Path $path) {
            Start-Process -FilePath $path -ArgumentList $url
            $chromeFound = $true
            Write-Host "Chrome opened successfully!" -ForegroundColor Green
            break
        }
    }
    
    if (-not $chromeFound) {
        # Fallback: use default browser
        Write-Host "Chrome not found, opening default browser..." -ForegroundColor Yellow
        Start-Process $url
    }
} catch {
    Write-Host "Error opening browser: $_" -ForegroundColor Red
    Write-Host "Please manually open: $url" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Dashboard URL: $url" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Keep the script running and wait for Streamlit
try {
    $streamlitProcess.WaitForExit()
} catch {
    Write-Host "Server stopped." -ForegroundColor Yellow
}
