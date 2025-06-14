"""
Setup script for AirSense - AI-Powered Air Quality Monitoring & Forecasting
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages."""
    try:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "setup_requirements.txt"
        ])
        print("All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'exports', 'config']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def verify_installation():
    """Verify that all required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'sklearn', 'xgboost', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("All required packages are installed!")
    return True

def main():
    """Main setup function."""
    print("=" * 60)
    print("AirSense Setup - AI-Powered Air Quality Monitoring")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Verify installation
    if not verify_installation():
        print("Setup completed with warnings. Some packages may need manual installation.")
    else:
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nTo run the application:")
        print("  streamlit run app.py")
        print("\nThe application will open in your browser at http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)