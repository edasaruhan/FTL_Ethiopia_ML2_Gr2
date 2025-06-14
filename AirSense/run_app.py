#!/usr/bin/env python3
"""
Quick start script for AirSense Application
This script handles common startup issues and provides better error messages.
"""

import sys
import os
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_package(package_name):
    """Install a missing package."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_requirements():
    """Check and install missing requirements."""
    requirements = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'plotly': 'plotly',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }
    
    missing = []
    for package, import_name in requirements.items():
        if not check_package(package, import_name):
            missing.append(package)
    
    if missing:
        print(f"Missing packages detected: {', '.join(missing)}")
        print("Installing missing packages...")
        
        for package in missing:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def create_directories():
    """Create necessary directories."""
    dirs = ['models', 'exports', 'config', 'data']
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def run_streamlit():
    """Run the Streamlit application."""
    try:
        # Try to run on default port first
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        print("Trying alternative port...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502"])
        except Exception as e2:
            print(f"Failed to start on alternative port: {e2}")

def main():
    """Main execution function."""
    print("AirSense - AI-Powered Air Quality Monitoring")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"Error: Python 3.8+ required. Current: {sys.version}")
        return
    
    # Create directories
    create_directories()
    
    # Check and install requirements
    if not check_and_install_requirements():
        print("Failed to install all requirements. Please run:")
        print("pip install -r setup_requirements.txt")
        return
    
    print("\nStarting AirSense application...")
    print("The app will open in your browser automatically.")
    print("If it doesn't open, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    print("-" * 50)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()