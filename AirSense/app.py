import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import custom modules
from data_handler import DataHandler
from model_trainer import ModelTrainer
from forecaster import Forecaster
from health_alerts import HealthAlerts
from visualizations import Visualizations
from utils import Utils

# Page configuration
st.set_page_config(
    page_title="AirSense - AI Air Quality Monitor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E8B57;
    margin-bottom: 2rem;
    animation: fadeInDown 1.2s ease-out;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #FFE4E1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #DC143C;
    }
    .alert-moderate {
        background-color: #FFF8DC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
    }
    .alert-good {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
    }
</style>
""", unsafe_allow_html=True)
theme = st.sidebar.selectbox(" Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
        body, .stApp {
            background-color: #111 !important;
            color: #eee !important;
        }
        .metric-card, .alert-good, .alert-moderate, .alert-high {
            background-color: #222 !important;
            color: #fff !important;
        }
        .main-header {
            color: #90ee90 !important;
        }
        .sub-header {
            color: #89CFF0 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🌬️ AIRSENSE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Air Quality Monitoring & Forecasting</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    
    # Initialize components
    data_handler = DataHandler()
    model_trainer = ModelTrainer()
    forecaster = Forecaster()
    health_alerts = HealthAlerts()
    visualizations = Visualizations()
    utils = Utils()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Select Function", [
        "🏠 Data Hub",
        "🔍 Data Exploration", 
        "🤖 AI Model Training",
        "🔮 AI Forecasting",
        "🚨 Health Alerts"
    ])
    
    if tab == "🏠 Data Hub":
        data_hub_page(data_handler)
    elif tab == "🔍 Data Exploration":
        data_exploration_page(visualizations)
    elif tab == "🤖 AI Model Training":
        model_training_page(model_trainer)
    elif tab == "🔮 AI Forecasting":
        forecasting_page(forecaster)
    elif tab == "🚨 Health Alerts":
        health_alerts_page(health_alerts)

def data_hub_page(data_handler):
    st.markdown('<h2 class="sub-header">🏠 Data Hub</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Custom Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    data = data_handler.load_uploaded_data(uploaded_file)
                    st.session_state.data = data
                    st.success(f"Data loaded successfully! Shape: {data.shape}")
                    st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with col2:
        st.subheader("Use Demo Dataset")
        if st.button("Load Beijing Demo Dataset"):
            try:
                with st.spinner("Loading Beijing demo data..."):
                    data = data_handler.load_demo_data()
                    st.session_state.data = data
                    st.success(f"Beijing demo data loaded! Shape: {data.shape}")
                    st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading demo data: {str(e)}")
    
    # Data info
    if st.session_state.data is not None:
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Features", len(st.session_state.data.columns))
        with col3:
            missing_values = st.session_state.data.isnull().sum().sum()
            st.metric("Missing Values", missing_values)

def data_exploration_page(visualizations):
    st.markdown('<h2 class="sub-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load data first from the Data Hub.")
        return
    
    data = st.session_state.data
    
    # Visualization options
    viz_type = st.selectbox("Choose Visualization", [
        "Time Series Trends",
        "Correlation Matrix",
        "Distribution Analysis",
        "Seasonal Patterns",
        "Diurnal Patterns"
    ])
    
    if viz_type == "Time Series Trends":
        fig = visualizations.plot_time_series(data)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix":
        fig = visualizations.plot_correlation_matrix(data)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribution Analysis":
        pollutant = st.selectbox("Select Pollutant", visualizations.get_pollutant_columns(data))
        fig = visualizations.plot_distribution(data, pollutant)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Seasonal Patterns":
        pollutant = st.selectbox("Select Pollutant", visualizations.get_pollutant_columns(data))
        fig = visualizations.plot_seasonal_patterns(data, pollutant)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Diurnal Patterns":
        pollutant = st.selectbox("Select Pollutant", visualizations.get_pollutant_columns(data))
        fig = visualizations.plot_diurnal_patterns(data, pollutant)
        st.plotly_chart(fig, use_container_width=True)

def model_training_page(model_trainer):
    st.markdown('<h2 class="sub-header">🤖 AI Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load data first from the Data Hub.")
        return
    
    data = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_pollutant = st.selectbox("Target Pollutant", model_trainer.get_pollutant_columns(data))
        model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost"])
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100)
    
    if st.button("Train Model"):
        try:
            with st.spinner("Training model..."):
                model, metrics, feature_importance = model_trainer.train_model(
                    data, target_pollutant, model_type, test_size, n_estimators
                )
                
                # Store trained model
                st.session_state.trained_models[f"{model_type}_{target_pollutant}"] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': feature_importance,
                    'target': target_pollutant
                }
                
                st.success("Model trained successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                
                # Feature importance plot
                if feature_importance is not None:
                    fig = model_trainer.plot_feature_importance(feature_importance)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    # Display trained models
    if st.session_state.trained_models:
        st.subheader("Trained Models")
        for model_name, model_info in st.session_state.trained_models.items():
            with st.expander(f"{model_name}"):
                metrics = model_info['metrics']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.4f}")

def forecasting_page(forecaster):
    st.markdown('<h2 class="sub-header">🔮 AI Forecasting</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("Please train a model first from the AI Model Training section.")
        return
    
    # Model selection
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Trained Model", model_names)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_steps = st.slider("Forecast Steps (hours)", 1, 72, 24)
    
    with col2:
        forecast_interval = st.selectbox("Forecast Interval", ["1H", "3H", "6H", "12H"])
    
    if st.button("Generate Forecast"):
        try:
            with st.spinner("Generating forecast..."):
                model_info = st.session_state.trained_models[selected_model]
                
                forecast_data = forecaster.generate_forecast(
                    st.session_state.data,
                    model_info['model'],
                    model_info['target'],
                    forecast_steps,
                    forecast_interval
                )
                
                st.session_state.forecasts[selected_model] = forecast_data
                
                # Plot forecast
                fig = forecaster.plot_forecast(
                    st.session_state.data,
                    forecast_data,
                    model_info['target']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data
                st.subheader("Forecast Data")
                st.dataframe(forecast_data)
                
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

def health_alerts_page(health_alerts):
    st.markdown('<h2 class="sub-header">🚨 Health Alerts</h2>', unsafe_allow_html=True)
    
    if not st.session_state.forecasts:
        st.warning("Please generate forecasts first from the AI Forecasting section.")
        return
    
    # Select forecast for health alerts
    forecast_names = list(st.session_state.forecasts.keys())
    selected_forecast = st.selectbox("Select Forecast", forecast_names)
    
    if selected_forecast:
        forecast_data = st.session_state.forecasts[selected_forecast]
        model_info = st.session_state.trained_models[selected_forecast]
        target_pollutant = model_info['target']
        
        # Generate health alerts
        alerts = health_alerts.generate_alerts(forecast_data, target_pollutant)
        
        # Display current alert
        current_alert = alerts[0] if alerts else None
        if current_alert:
            alert_class = f"alert-{current_alert['level'].lower()}"
            st.markdown(f"""
            <div class="{alert_class}">
                <h3>🚨 Current Alert: {current_alert['level']}</h3>
                <p><strong>AQI:</strong> {current_alert['aqi']}</p>
                <p><strong>Health Impact:</strong> {current_alert['health_impact']}</p>
                <p><strong>Recommendations:</strong> {current_alert['recommendations']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display all alerts
        st.subheader("Forecast Alert Timeline")
        
        for alert in alerts:
            alert_class = f"alert-{alert['level'].lower()}"
            st.markdown(f"""
            <div class="{alert_class}" style="margin-bottom: 1rem;">
                <h4>{alert['timestamp']} - {alert['level']}</h4>
                <p><strong>AQI:</strong> {alert['aqi']} | <strong>Value:</strong> {alert['value']:.2f} μg/m³</p>
                <p><strong>Health Impact:</strong> {alert['health_impact']}</p>
                <p><strong>Recommendations:</strong> {alert['recommendations']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert statistics
        st.subheader("Alert Statistics")
        alert_counts = {}
        for alert in alerts:
            level = alert['level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        cols = st.columns(len(alert_counts))
        for i, (level, count) in enumerate(alert_counts.items()):
            with cols[i]:
                st.metric(f"{level} Alerts", count)

if __name__ == "__main__":
    main()
