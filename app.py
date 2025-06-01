import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import traceback
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
# Import modules with proper error handling
try:
    from data_processor import DataProcessor
    from ml_models import AirQualityPredictor
    from visualizations import Visualizer
    from health_alerts import HealthAlertSystem
    from utils import load_sample_data, format_metrics
except ImportError as e:
    st.error(f"Missing required module: {e}")
    st.stop()
# Page Configuration
st.set_page_config(
    page_title="AirSense - AI Air Quality Forecasting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .ai-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card h4 {
        margin: 0;
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .metric-card h2 {
        margin: 0.5rem 0;
        color: #667eea;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .info-box {
        background: #e7f3ff;
        color: #0056b3;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        margin: 1.5rem 0;
    }
    
    .waiting-box {
        background: #fff9e6;
        color: #856404;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px dashed #ffc107;
        margin: 2rem 0;
    }
    
    .status-indicator {
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0;
    }
    
    .status-connected { background: #d4edda; color: #155724; }
    .status-waiting { background: #fff3cd; color: #856404; }
    </style>
    """, unsafe_allow_html=True)
# Initialize Session State
def initialize_session_state():
    defaults = {
        "data_loaded": False,
        "models_trained": False,
        "forecast_generated": False,
        "data": None,
        "processed_data": None,
        "model_results": None,
        "forecast_data": None,
        "target_pollutant": None,
    }
    
    try:
        defaults["data_processor"] = DataProcessor()
        os.makedirs("./airsense_models", exist_ok=True)
        defaults["predictor"] = AirQualityPredictor(model_dir="./airsense_models")
        defaults["visualizer"] = Visualizer()
        defaults["health_system"] = HealthAlertSystem()
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        defaults.update({
            "data_processor": None,
            "predictor": None,
            "visualizer": None,
            "health_system": None
        })
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
# UI Helper Functions
def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="ai-badge">AI-POWERED</div>
        <h1>🌬️ AirSense</h1>
        <p>Advanced Air Quality Forecasting & Health Alert System</p>
        <p>Urban Environmental Intelligence • ML Predictions • Health Alerts</p>
    </div>
    """, unsafe_allow_html=True)
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>🌬️ AirSense</h2>
            <p>Navigation Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        page_options = {
            "Data Hub": "data_upload_page",
            "Explore Data": "data_exploration_page",
            "Train Models": "model_training_page",
            "Forecast": "forecasting_page",
            "Health Alerts": "health_alerts_page"
        }
        selected_page_display = st.selectbox(
            "Select Module:",
            list(page_options.keys()),
            label_visibility="collapsed"
        )
        st.markdown("---")
        
        # System Status
        st.markdown("### System Status")
        
        data_status = "Connected" if st.session_state.data_loaded else "Waiting for data"
        models_status = "Trained" if st.session_state.models_trained else "Not trained"
        forecast_status = "Generated" if st.session_state.forecast_generated else "Not generated"
        
        status_class_data = "status-connected" if st.session_state.data_loaded else "status-waiting"
        status_class_models = "status-connected" if st.session_state.models_trained else "status-waiting"
        status_class_forecast = "status-connected" if st.session_state.forecast_generated else "status-waiting"
        
        st.markdown(f'<div class="status-indicator {status_class_data}">Data: {data_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-indicator {status_class_models}">Models: {models_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-indicator {status_class_forecast}">Forecast: {forecast_status}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Developed by Manus AI\n\nStart with the Data Hub to load sample data!")
    return page_options[selected_page_display]
def render_page_title(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f"*{subtitle}*")
    st.markdown("---")
def render_metric(label, value, help_text=""):
    st.markdown(f"""
    <div class="metric-card" title="{help_text}">
        <h4>{label}</h4>
        <h2>{value}</h2>
        <p>{help_text}</p>
    </div>
    """, unsafe_allow_html=True)
def render_info_box(content, icon="ℹ️"):
    st.markdown(f'<div class="info-box">{icon} {content}</div>', unsafe_allow_html=True)
def render_waiting_message(title, message, icon="⏳"):
    st.markdown(f"""
    <div class="waiting-box">
        <h3>{icon} {title}</h3>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)
# Page Implementations
def data_upload_page():
    render_page_title("Data Hub", "Upload your air quality data or use our demo dataset to get started.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Select a CSV file:",
            type="csv",
            help="Upload a CSV file containing hourly air quality and weather data."
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing your data..."):
                    raw_data = pd.read_csv(uploaded_file)
                    if st.session_state.data_processor:
                        st.session_state.processed_data = st.session_state.data_processor.load_and_preprocess(raw_data)
                        st.session_state.data_loaded = True
                        st.session_state.models_trained = False
                        st.session_state.forecast_generated = False
                        st.session_state.model_results = None
                        st.session_state.forecast_data = None
                        st.success("Data loaded and preprocessed successfully!")
                        st.rerun()
                    else:
                        st.error("Data processor not available")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                st.session_state.data_loaded = False
    with col2:
        st.subheader("Use Demo Data")
        st.write("Explore AirSense features using a sample Beijing dataset.")
        if st.button("Load Demo Dataset", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading and processing demo data..."):
                    demo_data_raw = load_sample_data()
                    if demo_data_raw.empty:
                        raise ValueError("Failed to load demo data.")
                    if st.session_state.data_processor:
                        st.session_state.processed_data = st.session_state.data_processor.load_and_preprocess(demo_data_raw)
                        st.session_state.data_loaded = True
                        st.session_state.models_trained = False
                        st.session_state.forecast_generated = False
                        st.session_state.model_results = None
                        st.session_state.forecast_data = None
                        st.success("Demo data loaded and preprocessed successfully!")
                        st.rerun()
                    else:
                        st.error("Data processor not available")
            except Exception as e:
                st.error(f"Error loading demo data: {e}")
                st.session_state.data_loaded = False
    st.markdown("---")
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        st.subheader("Data Overview")
        data = st.session_state.processed_data
        
        cols = st.columns(4)
        with cols[0]: 
            render_metric("Total Records", f"{len(data):,}", "Hourly data points")
        with cols[1]:
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                date_range = f"{data.index.min():%Y-%m-%d} to {data.index.max():%Y-%m-%d}"
            else:
                date_range = "Date range not available"
            render_metric("Date Range", date_range, "Coverage of the data")
        with cols[2]: 
            render_metric("Features", f"{len(data.columns)}", "Original & engineered features")
        with cols[3]:
            missing_pct = (data.isnull().sum().sum() / data.size) * 100
            render_metric("Missing Data", f"{missing_pct:.1f}%", "After preprocessing")
        st.subheader("Processed Data Sample")
        st.dataframe(data.head(10), use_container_width=True)
        with st.expander("Show Data Quality Summary"):
            try:
                quality_df = pd.DataFrame({
                    "Column": data.columns,
                    "Data Type": data.dtypes.astype(str),
                    "Missing Values": data.isnull().sum(),
                    "Missing %": (data.isnull().sum() / len(data) * 100).round(2)
                }).reset_index(drop=True)
                st.dataframe(quality_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate data quality summary: {e}")
    else:
        render_info_box("Please upload a dataset or load the demo data to proceed.")
def data_exploration_page():
    render_page_title("Explore Data", "Visualize patterns and relationships in the air quality data.")
    if not st.session_state.data_loaded or st.session_state.processed_data is None:
        render_waiting_message("Data Required", "Please load data via the Data Hub page first.")
        return
    data = st.session_state.processed_data
    visualizer = st.session_state.visualizer
    
    if not visualizer:
        st.error("Visualizer not available")
        return
    
    pollutant_cols = [col for col in data.columns if col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]]
    weather_cols = [col for col in data.columns if col in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]]
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Correlations", "Distributions", "Patterns"])
    with tab1:
        st.subheader("Pollutant Trends Over Time")
        if pollutant_cols:
            selected_ts = st.multiselect(
                "Select pollutants:", pollutant_cols,
                default=pollutant_cols[:min(3, len(pollutant_cols))],
                key="ts_multiselect"
            )
            if selected_ts:
                try:
                    fig = visualizer.create_time_series_plot(data, selected_ts)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating time series plot: {e}")
            else: 
                st.info("Select one or more pollutants to plot.")
        else:
            st.info("No standard pollutant columns found in the data.")
    with tab2:
        st.subheader("Correlation Matrix")
        default_corr_cols = pollutant_cols + weather_cols
        available_numeric_cols = list(data.select_dtypes(include=np.number).columns)
        
        if available_numeric_cols:
            selected_corr = st.multiselect(
                "Select features for correlation analysis:",
                available_numeric_cols,
                default=[col for col in default_corr_cols if col in available_numeric_cols][:8],
                key="corr_multiselect"
            )
            if len(selected_corr) >= 2:
                try:
                    fig = visualizer.create_correlation_matrix(data, selected_corr)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {e}")
            else: 
                st.info("Select at least two features to see correlations.")
        else:
            st.info("No numeric columns available for correlation analysis.")
    with tab3:
        st.subheader("Pollutant Distributions")
        if pollutant_cols:
            selected_dist = st.selectbox("Select pollutant:", pollutant_cols, key="dist_select")
            if selected_dist:
                try:
                    fig = visualizer.create_distribution_plot(data, selected_dist)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating distribution plot: {e}")
        else: 
            st.info("No standard pollutant columns found for distribution analysis.")
    with tab4:
        st.subheader("Temporal Patterns")
        if pollutant_cols:
            selected_pattern = st.selectbox("Select pollutant:", pollutant_cols, key="pattern_select")
            pattern_type = st.radio("Analyze by:", ["Monthly Seasonality", "Daily Patterns"], horizontal=True, key="pattern_type")
            if selected_pattern:
                try:
                    if "Monthly" in pattern_type:
                        fig = visualizer.create_seasonal_analysis(data, selected_pattern)
                    else:
                        fig = visualizer.create_daily_pattern_plot(data, selected_pattern)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating pattern plot: {e}")
        else:
            st.info("No standard pollutant columns found for pattern analysis.")
def model_training_page():
    render_page_title("Train Models", "Train machine learning models for air quality prediction.")
    if not st.session_state.data_loaded or st.session_state.processed_data is None:
        render_waiting_message("Data Required", "Please load data via the Data Hub page first.")
        return
    if not st.session_state.predictor:
        st.error("Predictor not available")
        return
    data = st.session_state.processed_data
    predictor = st.session_state.predictor
    
    pollutant_cols = [col for col in data.columns if col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]]
    
    if not pollutant_cols:
        st.warning("No standard pollutant columns found in the data for modeling.")
        return
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Training Configuration")
        
        target_pollutant = st.selectbox("Select target pollutant:", pollutant_cols)
        st.session_state.target_pollutant = target_pollutant
        
        model_types = st.multiselect(
            "Select models to train:",
            ["Random Forest", "XGBoost"],
            default=["Random Forest"]
        )
        
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("Start Training", type="primary", use_container_width=True):
            if model_types:
                try:
                    with st.spinner("Training models... This may take a few minutes."):
                        # Prepare features and target
                        X, y = predictor.prepare_features_target(data, target_pollutant)
                        
                        # Train models
                        results = predictor.train_models(X, y, target_pollutant, model_types, test_size)
                        st.session_state.model_results = results
                        st.session_state.models_trained = True
                        st.session_state.forecast_generated = False
                    st.success("Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error training models: {e}")
                    st.session_state.models_trained = False
            else:
                st.warning("Please select at least one model to train.")
    with col2:
        st.subheader("Training Results")
        
        if st.session_state.models_trained and st.session_state.model_results:
            results = st.session_state.model_results
            
            # Display metrics table
            metrics_data = []
            for model_name, model_data in results.items():
                metrics_data.append({
                    "Model": model_name,
                    "RMSE": f"{model_data['rmse']:.3f}",
                    "MAE": f"{model_data['mae']:.3f}",
                    "R²": f"{model_data['r2']:.3f}",
                    "CV Mean": f"{model_data['cv_mean']:.3f}",
                    "CV Std": f"{model_data['cv_std']:.3f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model info
            best_model = min(results.keys(), key=lambda k: results[k]["rmse"])
            st.success(f"Best Model: {best_model} (RMSE = {results[best_model]['rmse']:.3f})")
            
            # Feature importance
            if results[best_model].get('feature_importance') is not None:
                st.subheader("Feature Importance")
                importance_scores = results[best_model]['feature_importance']
                feature_names = results[best_model]['feature_names']
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores
                }).sort_values('importance', ascending=False).head(10)
                
                fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                           title="Top 10 Most Important Features")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            render_info_box("Train models to see results here.")
def forecasting_page():
    render_page_title("Forecast", "Generate air quality predictions using trained models.")
    if not st.session_state.data_loaded or st.session_state.processed_data is None:
        render_waiting_message("Data Required", "Please load data via the Data Hub page first.")
        return
    if not st.session_state.models_trained or not st.session_state.model_results:
        render_waiting_message("Models Required", "Please train models via the Train Models page first.")
        return
    if not st.session_state.predictor:
        st.error("Predictor not available")
        return
    predictor = st.session_state.predictor
    target_pollutant = st.session_state.target_pollutant
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Forecast Settings")
        
        available_models = list(st.session_state.model_results.keys())
        selected_model = st.selectbox("Select model:", available_models)
        
        forecast_hours = st.slider("Forecast horizon (hours):", 1, 72, 24)
        
        if st.button("Generate Forecast", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating forecast..."):
                    forecast_data = predictor.generate_forecast(
                        st.session_state.processed_data, 
                        target_pollutant,
                        selected_model,
                        forecast_hours
                    )
                    st.session_state.forecast_data = forecast_data
                    st.session_state.forecast_generated = True
                st.success("Forecast generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.session_state.forecast_generated = False
    with col2:
        st.subheader("Forecast Results")
        
        if st.session_state.forecast_generated and st.session_state.forecast_data is not None:
            forecast_data = st.session_state.forecast_data
            
            # Create forecast visualization
            try:
                if st.session_state.visualizer:
                    fig = st.session_state.visualizer.create_forecast_plot(
                        st.session_state.processed_data,
                        forecast_data,
                        target_pollutant
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.subheader("Forecast Values")
                st.dataframe(forecast_data.head(24), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying forecast: {e}")
        else:
            render_info_box("Generate a forecast to see predictions here.")
def health_alerts_page():
    render_page_title("Health Alerts", "Get health recommendations based on air quality levels.")
    if not st.session_state.health_system:
        st.error("Health alert system not available")
        return
    health_system = st.session_state.health_system
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        st.subheader("Current Air Quality Status")
        
        try:
            latest_data = data.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            pollutants = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
            available_pollutants = [p for p in pollutants if p in data.columns]
            
            for i, pollutant in enumerate(available_pollutants[:3]):
                with [col1, col2, col3][i]:
                    value = latest_data[pollutant]
                    
                    # Calculate AQI
                    aqi = health_system._calculate_aqi(value, pollutant)
                    alert_level = health_system.get_aqi_category(aqi)
                    
                    color_map = {
                        "Good": "🟢",
                        "Moderate": "🟡", 
                        "Unhealthy for Sensitive Groups": "🟠",
                        "Unhealthy": "🔴",
                        "Very Unhealthy": "🟣",
                        "Hazardous": "🔴"
                    }
                    
                    status_icon = color_map.get(alert_level, "⚪")
                    
                    st.metric(
                        label=f"{status_icon} {pollutant}",
                        value=f"{value:.1f}",
                        help=f"Level: {alert_level}"
                    )
            
            # Health recommendations
            st.subheader("Health Recommendations")
            
            if available_pollutants:
                worst_pollutant = max(available_pollutants, 
                                    key=lambda p: latest_data[p] if pd.notna(latest_data[p]) else 0)
                worst_value = latest_data[worst_pollutant]
                
                aqi = health_system._calculate_aqi(worst_value, worst_pollutant)
                alert_level = health_system.get_aqi_category(aqi)
                recommendations = health_system.get_health_recommendations(alert_level)
                
                if recommendations:
                    st.info(f"**{alert_level}**: {recommendations.get('General Public', 'No specific recommendations available.')}")
            
        except Exception as e:
            st.error(f"Error processing health alerts: {e}")
    
    else:
        render_info_box("Load data to see current air quality status and health recommendations.")
    
    # Show forecast alerts if available
    if st.session_state.forecast_generated and st.session_state.forecast_data is not None:
        st.subheader("Forecast Health Alerts")
        
        try:
            target_pollutant = st.session_state.target_pollutant
            forecast_series = st.session_state.forecast_data.iloc[:, 0]  # First column
            
            alerts_df = health_system.generate_alerts(forecast_series, target_pollutant)
            
            if not alerts_df.empty:
                # Summary
                summary = health_system.generate_alert_summary(alerts_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Hours", summary['total_hours'])
                with col2:
                    st.metric("Max AQI", f"{summary['max_aqi']:.0f}" if not pd.isna(summary['max_aqi']) else "N/A")
                with col3:
                    st.metric("Unhealthy Hours", summary['hours_unhealthy_or_worse'])
                
                # Alert distribution
                st.subheader("Alert Level Distribution")
                if summary['alert_distribution']:
                    alert_dist_df = pd.DataFrame(list(summary['alert_distribution'].items()), 
                                               columns=['Alert Level', 'Hours'])
                    fig = px.pie(alert_dist_df, values='Hours', names='Alert Level',
                               title="Distribution of Alert Levels in Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed alerts
                with st.expander("View Detailed Alerts"):
                    st.dataframe(alerts_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error generating forecast alerts: {e}")
    
    # General health information
    with st.expander("Air Quality Index (AQI) Information"):
        st.markdown("""
        **Air Quality Levels:**
        - 🟢 **Good (0-50)**: Air quality is satisfactory
        - 🟡 **Moderate (51-100)**: Acceptable for most people
        - 🟠 **Unhealthy for Sensitive Groups (101-150)**: Sensitive individuals may experience problems
        - 🔴 **Unhealthy (151-200)**: Everyone may experience problems
        - 🟣 **Very Unhealthy (201-300)**: Health warnings for everyone
        - 🔴 **Hazardous (301+)**: Emergency conditions
        """)
# Main Application
def main():
    try:
        load_css()
        initialize_session_state()
        
        render_header()
        
        selected_page = render_sidebar()
        
        if selected_page == "data_upload_page":
            data_upload_page()
        elif selected_page == "data_exploration_page":
            data_exploration_page()
        elif selected_page == "model_training_page":
            model_training_page()
        elif selected_page == "forecasting_page":
            forecasting_page()
        elif selected_page == "health_alerts_page":
            health_alerts_page()
        else:
            st.error(f"Unknown page: {selected_page}")
            
    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)
if __name__ == "__main__":
    main()
# Key fixes I made:
# Removed all problematic symbols from the code (quotes, special characters)
# Simplified the UI with a single logo in the header
# Fixed import issues with proper error handling
# Corrected session state management
# Fixed CSS issues and streamlined styling
# Improved navigation and removed complex icons
# Added proper error handling throughout
# Fixed data processing pipeline
# Corrected method calls in all modules
# Added status indicators in sidebar