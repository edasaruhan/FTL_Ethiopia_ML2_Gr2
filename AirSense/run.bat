@echo off
title AirSense - AI Air Quality Monitor
echo ================================================
echo AirSense - AI-Powered Air Quality Monitoring
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python found. Checking dependencies...

REM Create directories if they don't exist
if not exist "models" mkdir models
if not exist "exports" mkdir exports
if not exist "config" mkdir config
if not exist "data" mkdir data

REM Install requirements
echo Installing required packages...
python -m pip install -q streamlit pandas numpy plotly scikit-learn xgboost joblib

if errorlevel 1 (
    echo.
    echo Warning: Some packages may not have installed correctly
    echo Trying alternative installation...
    python -m pip install --user streamlit pandas numpy plotly scikit-learn xgboost joblib
)

echo.
echo Starting AirSense application...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ================================================

REM Start the application
python -m streamlit run app.py --server.port 8501

if errorlevel 1 (
    echo.
    echo Failed to start on port 8501, trying port 8502...
    python -m streamlit run app.py --server.port 8502
)

pause