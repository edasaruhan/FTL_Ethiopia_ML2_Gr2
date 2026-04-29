@echo off
echo AirSense System Startup Script
echo ==============================
echo.

REM Try different Python installations
echo Checking for Python installations...

REM Try standard python command
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    python --version
    echo.
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo.
    echo Starting AirSense system...
    python src/main.py all
    goto :end
)

REM Try py command
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    py --version
    echo.
    echo Installing dependencies...
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    echo.
    echo Starting AirSense system...
    py src/main.py all
    goto :end
)

REM Try PostgreSQL Python
"C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe" --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe" --version
    echo.
    echo Installing dependencies...
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe" -m pip install --upgrade pip
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe" -m pip install -r requirements.txt
    echo.
    echo Starting AirSense system...
    "C:\Program Files\PostgreSQL\18\pgAdmin 4\python.exe" src/main.py all
    goto :end
)

REM Try Nmap Python
"C:\Program Files (x86)\Nmap\zenmap\bin\python.exe" --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python: 
    "C:\Program Files (x86)\Nmap\zenmap\bin\python.exe" --version
    echo.
    echo Installing dependencies...
    "C:\Program Files (x86)\Nmap\zenmap\bin\python.exe" -m pip install --upgrade pip
    "C:\Program Files (x86)\Nmap\zenmap\bin\python.exe" -m pip install -r requirements.txt
    echo.
    echo Starting AirSense system...
    "C:\Program Files (x86)\Nmap\zenmap\bin\python.exe" src/main.py all
    goto :end
)

echo No Python installation found!
echo Please install Python 3.8+ from https://python.org
echo.
echo After installing Python, run this script again.
echo.

:end
pause
