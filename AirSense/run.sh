#!/bin/bash

# AirSense - AI-Powered Air Quality Monitoring
# Unix/Linux/macOS startup script

echo "================================================"
echo "AirSense - AI-Powered Air Quality Monitoring"
echo "================================================"
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

echo "Python 3 found. Checking dependencies..."

# Create directories if they don't exist
mkdir -p models exports config data

# Function to install packages
install_packages() {
    echo "Installing required packages..."
    python3 -m pip install -q streamlit pandas numpy plotly scikit-learn xgboost joblib
    
    if [ $? -ne 0 ]; then
        echo
        echo "Warning: Some packages may not have installed correctly"
        echo "Trying alternative installation..."
        python3 -m pip install --user streamlit pandas numpy plotly scikit-learn xgboost joblib
    fi
}

# Install packages
install_packages

echo
echo "Starting AirSense application..."
echo "The app will open in your browser at http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo "================================================"

# Start the application
python3 -m streamlit run app.py --server.port 8501

# If port 8501 fails, try 8502
if [ $? -ne 0 ]; then
    echo
    echo "Failed to start on port 8501, trying port 8502..."
    python3 -m streamlit run app.py --server.port 8502
fi