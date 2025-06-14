# AirSense - AI-Powered Air Quality Monitoring & Forecasting 🌬️

AirSense is a comprehensive Streamlit web application designed to monitor, analyze, and forecast urban air quality using machine learning. It provides actionable health alerts based on predicted pollution levels, empowering users to make informed decisions.

## ✨ Features

- **🏠 Data Hub**: Upload custom air quality CSV data or use the built-in Beijing demo dataset
- **🔍 Data Exploration**: Interactive visualizations for pollutant trends, correlations, distributions, and temporal patterns
- **🤖 AI Model Training**: Train Random Forest and XGBoost models to predict specific pollutants (PM2.5, NO₂, etc.)
- **🔮 AI Forecasting**: Generate multi-step ahead forecasts for selected pollutants using trained models
- **🚨 Health Alerts**: View AQI-based health alerts and recommendations derived from forecasts, aligned with EPA standards
- **🎨 Modern UI**: Clean, responsive, and visually appealing interface with professional styling

## 🎯 Project Goal

To provide an accessible tool for analyzing historical and real-time air quality data, forecasting future pollution levels, and delivering timely health alerts. This project contributes to:

- **SDG 3** (Good Health and Well-being): By addressing air pollution risks
- **SDG 11** (Sustainable Cities and Communities): By supporting informed urban planning and public awareness

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend & ML**: Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization**: Plotly
- **Model/Data Persistence**: Joblib

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation & Setup

### Step 1: Download the Project
Download all project files to your local directory.

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv airsense_env

# Activate virtual environment
# On Windows:
airsense_env\Scripts\activate
# On macOS/Linux:
source airsense_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r setup_requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
AirSense/
├── app.py                    # Main Streamlit application
├── data_handler.py          # Data loading and preprocessing
├── model_trainer.py         # Machine learning model training
├── forecaster.py            # Air quality forecasting
├── health_alerts.py         # Health alert generation
├── visualizations.py        # Data visualization functions
├── utils.py                 # Utility functions
├── setup_requirements.txt   # Python dependencies
├── README.md                # This file
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   └── beijing_demo.csv     # Demo dataset
├── models/                  # Saved ML models (created automatically)
├── exports/                 # Data exports (created automatically)
└── config/                  # App configurations (created automatically)
```

## 📊 Data Format

For custom data uploads, ensure your CSV file contains:

**Required columns:**
- `datetime` (or similar: date, time, timestamp)
- At least one pollutant: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`

**Optional columns:**
- `temperature`, `humidity`, `pressure`, `wind_speed`, `wind_direction`

**Example:**
```csv
datetime,PM2.5,PM10,NO2,SO2,CO,O3,temperature,humidity
2024-01-01 00:00:00,25.3,45.2,30.1,15.2,1.2,80.5,15.2,65.3
2024-01-01 01:00:00,28.7,48.9,32.5,16.8,1.4,75.2,14.8,67.1
```

## 🎮 How to Use

### 1. Data Hub
- Upload your own CSV file or load the Beijing demo dataset
- View data summary and statistics

### 2. Data Exploration
- Visualize time series trends
- Explore correlations between pollutants
- Analyze seasonal and diurnal patterns
- View distribution statistics

### 3. AI Model Training
- Select target pollutant to predict
- Choose between Random Forest and XGBoost algorithms
- Configure model parameters
- View training metrics and feature importance

### 4. AI Forecasting
- Select a trained model
- Configure forecast horizon (1-72 hours)
- Generate predictions with confidence intervals
- Visualize forecast results

### 5. Health Alerts
- View AQI-based health alerts
- Get detailed health recommendations
- See alert timeline and statistics
- Export alerts for further analysis

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Make sure all dependencies are installed
   pip install -r setup_requirements.txt
   ```

2. **Streamlit Command Not Found**
   ```bash
   # Ensure streamlit is installed
   pip install streamlit
   ```

3. **Port Already in Use**
   ```bash
   # Use a different port
   streamlit run app.py --server.port 8502
   ```

4. **Data Loading Issues**
   - Ensure CSV file has proper datetime column
   - Check for missing or invalid data values
   - Verify column names match expected format

5. **Model Training Errors**
   - Ensure sufficient data (at least 24 records recommended)
   - Check for missing values in target pollutant
   - Verify data types are numeric

## 📈 Performance Tips

- For large datasets (>10,000 records), consider sampling for faster visualization
- Use smaller forecast horizons for quicker predictions
- Regularly clean temporary files using the built-in utility functions

## 🔒 Data Privacy

- All data processing happens locally on your machine
- No data is sent to external servers
- Models and exports are saved locally in the project directory

## 🤝 Contributing

This project is designed for educational and research purposes. Feel free to modify and extend the functionality based on your specific needs.

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check this README for troubleshooting steps
2. Ensure all dependencies are properly installed
3. Verify your data format matches the requirements
4. Check Python and package versions compatibility

## 🌟 Version History

- **v1.0.0** - Initial release with core functionality
  - Data handling and preprocessing
  - Machine learning model training
  - Forecasting capabilities
  - Health alert system
  - Professional UI/UX

---

**Built with ❤️ for cleaner air and healthier communities**