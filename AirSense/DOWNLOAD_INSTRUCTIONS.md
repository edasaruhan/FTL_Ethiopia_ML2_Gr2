# Download Instructions for AirSense Project

## What You Get
A complete, ready-to-run AirSense application with all dependencies and setup scripts to avoid common installation issues.

## Download the Complete Project
The file `AirSense_Complete_Project.tar.gz` contains everything you need.

## Quick Start After Download

### For Windows Users:
1. Extract the downloaded file
2. Double-click `run.bat`
3. The application will start automatically

### For Mac/Linux Users:
1. Extract the downloaded file
2. Open terminal in the project folder
3. Run: `./run.sh`
4. The application will start automatically

### Manual Setup (If Scripts Don't Work):
1. Extract all files
2. Open command prompt/terminal in the project folder
3. Run: `python setup.py`
4. Then run: `streamlit run app.py`

## What's Included in the Package

### Core Application Files:
- `app.py` - Main Streamlit application
- `data_handler.py` - Data processing and validation
- `model_trainer.py` - AI model training functionality
- `forecaster.py` - Air quality forecasting
- `health_alerts.py` - Health alert system
- `visualizations.py` - Interactive charts and graphs
- `utils.py` - Utility functions

### Setup & Configuration:
- `setup_requirements.txt` - All required Python packages
- `setup.py` - Automated installation script
- `run_app.py` - Smart application launcher
- `run.bat` - Windows startup script
- `run.sh` - Mac/Linux startup script
- `.streamlit/config.toml` - Pre-configured Streamlit settings

### Documentation:
- `README.md` - Complete project documentation
- `INSTALLATION_GUIDE.md` - Detailed troubleshooting guide
- `DOWNLOAD_INSTRUCTIONS.md` - This file

### Sample Data:
- `data/beijing_demo.csv` - Real Beijing air quality data for testing

## System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection for initial package installation
- Modern web browser

## Troubleshooting Common Issues

### Issue: "Python not found"
**Solution:** Install Python from python.org, ensure it's added to PATH

### Issue: Package installation fails
**Solution:** Try the automated scripts first (`run.bat` or `run.sh`)

### Issue: Port already in use
**Solution:** The scripts automatically try alternative ports

### Issue: Permission errors
**Solution:** Run as administrator (Windows) or use `sudo` (Mac/Linux)

## Features Included
- Upload your own air quality data (CSV format)
- Interactive data exploration and visualization
- Train AI models (Random Forest, XGBoost)
- Generate air quality forecasts
- Receive health alerts based on EPA standards
- Professional, responsive web interface
- Data export capabilities

## Support
All common installation issues are handled automatically by the included scripts. The project is designed to work out-of-the-box on Windows, Mac, and Linux systems.

The application will run locally on your machine - no internet connection required after initial setup.