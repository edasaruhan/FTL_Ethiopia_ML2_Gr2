import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Configure page
st.set_page_config(
    page_title="AirSense Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE = "http://localhost:8000"

def fetch_api(endpoint):
    """Fetch data from API."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the backend is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def post_api(endpoint, data=None):
    """Post data to API."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error posting data: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #2E8B57;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #F0F8FF;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #2E8B57;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌬️ AirSense Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Air Quality Monitoring & Forecasting</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "📊 Dashboard",
        "📈 Data Explorer", 
        "🔮 Forecasts",
        "🚨 Health Alerts",
        "⚙️ Settings"
    ])
    
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📈 Data Explorer":
        data_explorer_page()
    elif page == "🔮 Forecasts":
        forecasts_page()
    elif page == "🚨 Health Alerts":
        health_alerts_page()
    elif page == "⚙️ Settings":
        settings_page()

def dashboard_page():
    """Main dashboard with overview metrics."""
    st.header("📊 Real-time Dashboard")
    
    # Check API health
    health = fetch_api("/health")
    if not health:
        st.warning("API is not available. Some features may not work.")
        return
    
    # Fetch current AQI
    aqi_data = fetch_api("/aqi")
    if aqi_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current AQI",
                f"{aqi_data['aqi']:.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Air Quality",
                aqi_data['level'],
                delta=None
            )
        
        with col3:
            latest_pm25 = aqi_data['component_aqi']['PM2.5']
            st.metric(
                "PM2.5 AQI",
                f"{latest_pm25:.0f}",
                delta=None
            )
        
        with col4:
            latest_pm10 = aqi_data['component_aqi']['PM10']
            st.metric(
                "PM10 AQI",
                f"{latest_pm10:.0f}",
                delta=None
            )
        
        # Health recommendations
        st.subheader("🏥 Health Recommendations")
        st.info(aqi_data['health_impact'])
        st.success(aqi_data['recommendations'])
    
    # Fetch recent data
    data = fetch_api("/data?limit=168")  # Last week
    if data and data['data']:
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Time series plot
        st.subheader("📈 24-Hour Trends")
        
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if available_pollutants:
            fig = go.Figure()
            
            for pollutant in available_pollutants[:3]:  # Show top 3
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[pollutant],
                    mode='lines',
                    name=pollutant,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Pollutant Concentrations (Last 24 Hours)",
                xaxis_title="Time",
                yaxis_title="Concentration (μg/m³)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def data_explorer_page():
    """Data exploration and analysis."""
    st.header("📈 Data Explorer")
    
    # Fetch data
    limit = st.slider("Number of records", 100, 5000, 1000)
    data = fetch_api(f"/data?limit={limit}")
    
    if not data or not data['data']:
        st.warning("No data available")
        return
    
    df = pd.DataFrame(data['data'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['datetime'].min().date())
    with col2:
        end_date = st.date_input("End Date", df['datetime'].max().date())
    
    # Filter data
    mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
    df_filtered = df[mask]
    
    # Display data
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Records", len(df_filtered))
    with col2:
        st.metric("Date Range", f"{(end_date - start_date).days} days")
    with col3:
        st.metric("Columns", len(df_filtered.columns))
    
    # Visualization options
    viz_type = st.selectbox("Visualization Type", [
        "Time Series",
        "Distribution",
        "Correlation Heatmap",
        "Box Plot"
    ])
    
    if viz_type == "Time Series":
        pollutant = st.selectbox("Select Pollutant", 
                               [col for col in df_filtered.columns if col not in ['datetime']])
        
        fig = px.line(df_filtered, x='datetime', y=pollutant, 
                     title=f"{pollutant} Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribution":
        pollutant = st.selectbox("Select Pollutant", 
                               [col for col in df_filtered.columns if col not in ['datetime']])
        
        fig = px.histogram(df_filtered, x=pollutant, 
                         title=f"{pollutant} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df_filtered[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation")
    
    elif viz_type == "Box Plot":
        pollutant = st.selectbox("Select Pollutant", 
                               [col for col in df_filtered.columns if col not in ['datetime']])
        
        df_filtered['hour'] = df_filtered['datetime'].dt.hour
        fig = px.box(df_filtered, x='hour', y=pollutant, 
                    title=f"{pollutant} by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Data Table")
    st.dataframe(df_filtered)

def forecasts_page():
    """Forecasting interface."""
    st.header("🔮 Air Quality Forecasts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_pollutant = st.selectbox("Target Pollutant", 
                                      ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])
    
    with col2:
        model_type = st.selectbox("Model Type", ["arima", "lstm"])
    
    with col3:
        forecast_steps = st.slider("Forecast Hours", 1, 72, 24)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            forecast_data = post_api("/forecast", {
                "target_pollutant": target_pollutant,
                "model_type": model_type,
                "steps": forecast_steps
            })
            
            if forecast_data:
                st.success(f"Forecast generated using {model_type.upper()} model")
                
                # Plot forecast
                forecast_df = pd.DataFrame(forecast_data['forecast'])
                forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_df['datetime'],
                    y=forecast_df['value'],
                    mode='lines+markers',
                    name=f'{target_pollutant} Forecast',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{target_pollutant} Forecast ({model_type.upper()})",
                    xaxis_title="Time",
                    yaxis_title="Concentration (μg/m³)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📋 Forecast Values")
                st.dataframe(forecast_df)
    
    # Available models
    st.subheader("🤖 Available Models")
    models = fetch_api("/models")
    if models:
        st.json(models)

def health_alerts_page():
    """Health alerts and recommendations."""
    st.header("🚨 Health Alerts")
    
    # Current AQI
    aqi_data = fetch_api("/aqi")
    if aqi_data:
        # Alert level styling
        aqi_level = aqi_data['level']
        
        if aqi_level == "Good":
            alert_color = "green"
        elif aqi_level == "Moderate":
            alert_color = "yellow"
        elif "Unhealthy" in aqi_level:
            alert_color = "red"
        else:
            alert_color = "orange"
        
        st.markdown(f"""
        <div style="background-color: {alert_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white;">Current Alert: {aqi_level}</h2>
            <p style="color: white; font-size: 18px;">AQI: {aqi_data['aqi']:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏥 Health Impact")
            st.info(aqi_data['health_impact'])
        
        with col2:
            st.subheader("💡 Recommendations")
            st.success(aqi_data['recommendations'])
    
    # Statistics
    st.subheader("📊 Air Quality Statistics")
    stats = fetch_api("/stats")
    if stats and 'statistics' in stats:
        stats_df = pd.DataFrame(stats['statistics']).T
        st.dataframe(stats_df)

def settings_page():
    """Settings and configuration."""
    st.header("⚙️ Settings")
    
    st.subheader("🔄 Pipeline Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Data Pipeline"):
            with st.spinner("Starting pipeline..."):
                result = post_api("/pipeline/run")
                if result:
                    st.success("Pipeline started successfully")
                else:
                    st.error("Failed to start pipeline")
    
    with col2:
        if st.button("Check API Health"):
            health = fetch_api("/health")
            if health:
                st.json(health)
    
    st.subheader("📊 System Information")
    
    # API status
    health = fetch_api("/health")
    if health:
        st.write("**API Status:**")
        st.json(health)

if __name__ == "__main__":
    main()
