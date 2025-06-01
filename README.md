# AirSense - AI-Powered Air Quality Monitoring & Forecasting 🌬️

AirSense is a Streamlit web application designed to monitor, analyze, and forecast urban air quality using machine learning. It provides actionable health alerts based on predicted pollution levels, empowering users to make informed decisions.




## ✨ Features

*   **🏠 Data Hub:** Upload custom air quality CSV data or use the built-in Beijing demo dataset.
*   **🔍 Data Exploration:** Interactively visualize pollutant trends, correlations, distributions, and temporal patterns (seasonal/diurnal) using Plotly.
*   **🤖 AI Model Training:** Train Random Forest and XGBoost models (or others) to predict specific pollutants (e.g., PM2.5, NO₂).
*   **🔮 AI Forecasting:** Generate multi-step ahead forecasts for selected pollutants using trained models.
*   **🚨 Health Alerts:** View AQI-based health alerts and recommendations derived from the forecast, aligned with EPA standards.
*   **🎨 Modern UI:** Clean, responsive, and visually appealing interface built with Streamlit and custom CSS.

## 🎯 Project Goal

To provide an accessible tool for analyzing historical and real-time air quality data, forecasting future pollution levels, and delivering timely health alerts. This project aims to contribute to:

*   **SDG 3 (Good Health and Well-being):** By addressing air pollution risks.
*   **SDG 11 (Sustainable Cities and Communities):** By supporting informed urban planning and public awareness.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Backend & ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost
*   **Visualization:** Plotly
*   **Model/Data Persistence:** Joblib

## 📁 Project Structure

```
airsense_project/
├── app.py                  # Main Streamlit application
├── data_processor.py       # Data loading and preprocessing logic
├── ml_models.py            # ML model training and forecasting logic
├── visualizations.py       # Plotly visualization functions
├── health_alerts.py        # AQI calculation and health alert logic
├── utils.py                # Utility functions (e.g., load sample data)
├── style.css               # Custom CSS for styling
├── requirements.txt        # Python dependencies
├── data/
│   └── sample_beijing_airquality.csv # Sample data file
├── airsense_models/        # Directory to store saved models and scalers (created automatically)
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+ installed.
*   `pip` (Python package installer).

### Installation

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Navigate to Directory:** Open your terminal or command prompt and change to the project directory:
    ```bash
    cd path/to/airsense_project
    ```
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Ensure you are in the project directory** (`airsense_project`) in your activated virtual environment.
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
3.  The application should automatically open in your default web browser.

## 📖 Usage

1.  **Data Hub:** Start by uploading your own hourly air quality data (CSV format, similar structure to the Beijing dataset) or click "Load Demo Dataset".
2.  **Explore Data:** Navigate through the tabs to visualize time series, correlations, distributions, and patterns in the loaded data.
3.  **Train Models:** Select a target pollutant (e.g., PM2.5) and choose the ML models (Random Forest, XGBoost) to train. Click "Start Training".
4.  **Forecast:** Once models are trained, select a model and specify the forecast horizon (in hours). Click "Generate Forecast".
5.  **Health Alerts:** After generating a forecast, view the corresponding health alerts, AQI levels, and recommendations based on the predicted pollutant concentrations.

## 📊 Data Format

The application expects input data (either uploaded or demo) in a CSV format with columns similar to the Beijing Multi-Site Air Quality dataset. Key expected columns include:

*   `year`, `month`, `day`, `hour` (or a single `datetime` column)
*   Pollutant columns (e.g., `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`)
*   Weather columns (e.g., `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`)

The `data_processor.py` module handles combining date/time columns, cleaning missing values (interpolation, fill), feature engineering (lags, rolling means, cyclical time features), and outlier capping.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details (if included).

---
