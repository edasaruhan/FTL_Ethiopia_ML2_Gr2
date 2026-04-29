@echo off
echo Starting AirSense Backend (API Server)
echo =====================================
echo.

REM Try different Python installations
echo Checking for Python installations...

python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    python --version
    echo.
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo.
    echo Starting Backend API Server...
    echo Backend will be available at: http://localhost:8000
    echo API Documentation: http://localhost:8000/docs
    echo.
    python src/main.py api
    goto :end
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    py --version
    echo.
    echo Installing dependencies...
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    echo.
    echo Starting Backend API Server...
    echo Backend will be available at: http://localhost:8000
    echo API Documentation: http://localhost:8000/docs
    echo.
    py src/main.py api
    goto :end
)

echo No Python installation found!
echo Please install Python 3.8+ from https://python.org

:end
pause
