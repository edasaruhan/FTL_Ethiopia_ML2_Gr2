# AirSense System Startup PowerShell Script
Write-Host "AirSense System Startup Script" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""

# Function to test Python installation
function Test-PythonInstallation {
    param($PythonPath)
    
    if (Test-Path $PythonPath) {
        try {
            $result = & $PythonPath --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Found Python: $result" -ForegroundColor Green
                return $true
            }
        } catch {
            # Python not working
        }
    }
    return $false
}

# Function to run AirSense with specific Python
function Run-AirSense {
    param($PythonPath)
    
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    try {
        & $PythonPath -m pip install --upgrade pip
        & $PythonPath -m pip install -r requirements.txt
        
        Write-Host "Starting AirSense system..." -ForegroundColor Yellow
        Write-Host "Dashboard will be available at: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "API Documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
        Write-Host ""
        
        & $PythonPath src/main.py all
    } catch {
        Write-Host "Error running AirSense: $_" -ForegroundColor Red
    }
}

# Try different Python installations
$pythonPaths = @(
    "python",
    "py", 
    "python3",
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe",
    "C:\Program Files (x86)\Nmap\zenmap\bin\python.exe"
)

foreach ($pythonPath in $pythonPaths) {
    Write-Host "Testing: $pythonPath" -ForegroundColor White
    if (Test-PythonInstallation -PythonPath $pythonPath) {
        Write-Host "Using Python: $pythonPath" -ForegroundColor Green
        Run-AirSense -PythonPath $pythonPath
        exit 0
    }
}

Write-Host "No working Python installation found!" -ForegroundColor Red
Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
Write-Host "After installing Python, run this script again." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
