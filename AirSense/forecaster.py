import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

class Forecaster:
    """Handles air quality forecasting using trained models."""
    
    def __init__(self):
        pass
    
    def generate_forecast(self, historical_data, model, target_pollutant, 
                         forecast_steps, forecast_interval='1H'):
        """Generate multi-step ahead forecasts."""
        try:
            # Prepare the most recent data for forecasting
            data = historical_data.copy()
            
            # Get the last datetime
            last_datetime = data['datetime'].max()
            
            # Create future datetime index
            freq_map = {'1H': 'H', '3H': '3H', '6H': '6H', '12H': '12H'}
            freq = freq_map.get(forecast_interval, 'H')
            
            future_dates = pd.date_range(
                start=last_datetime + pd.Timedelta(hours=1),
                periods=forecast_steps,
                freq=freq
            )
            
            # Initialize forecast array
            forecasts = []
            
            # Use the last window of data for iterative forecasting
            window_size = min(168, len(data))  # Use last week of data
            forecast_data = data.tail(window_size).copy()
            
            for i, future_date in enumerate(future_dates):
                # Prepare features for this forecast step
                features = self._prepare_forecast_features(
                    forecast_data, target_pollutant, future_date
                )
                
                # Make prediction
                prediction = model.predict([features])[0]
                
                # Ensure prediction is reasonable (non-negative)
                prediction = max(0, prediction)
                
                # Add prediction to the dataset for next iteration
                new_row = self._create_forecast_row(
                    forecast_data, future_date, target_pollutant, prediction
                )
                
                forecast_data = pd.concat([forecast_data, new_row], ignore_index=True)
                
                # Store forecast
                forecasts.append({
                    'datetime': future_date,
                    'forecast': prediction,
                    'step': i + 1
                })
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecasts)
            forecast_df['target_pollutant'] = target_pollutant
            
            return forecast_df
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")
    
    def _prepare_forecast_features(self, data, target_pollutant, forecast_datetime):
        """Prepare features for a single forecast step."""
        # Get the most recent data point
        latest_data = data.iloc[-1].copy()
        
        # Extract temporal features from forecast datetime
        hour = forecast_datetime.hour
        month = forecast_datetime.month
        dayofweek = forecast_datetime.dayofweek
        
        # Initialize features list
        features = []
        
        # Add other pollutants (use latest values)
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for col in pollutant_cols:
            if col != target_pollutant and col in data.columns:
                features.append(latest_data[col])
        
        # Add temporal features
        features.extend([
            hour,
            month,
            dayofweek,
            np.sin(2 * np.pi * hour / 24),  # hour_sin
            np.cos(2 * np.pi * hour / 24),  # hour_cos
            np.sin(2 * np.pi * month / 12), # month_sin
            np.cos(2 * np.pi * month / 12), # month_cos
            np.sin(2 * np.pi * dayofweek / 7), # day_sin
            np.cos(2 * np.pi * dayofweek / 7)  # day_cos
        ])
        
        # Add meteorological features if available
        meteo_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
        for col in meteo_cols:
            if col in data.columns:
                features.append(latest_data[col])
        
        # Add lagged features
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(data) > lag:
                lag_value = data[target_pollutant].iloc[-(lag+1)]
                features.append(lag_value)
            else:
                features.append(data[target_pollutant].mean())
        
        # Add rolling features
        for window in [3, 6, 12, 24]:
            if len(data) >= window:
                roll_value = data[target_pollutant].tail(window).mean()
                features.append(roll_value)
            else:
                features.append(data[target_pollutant].mean())
        
        return np.array(features)
    
    def _create_forecast_row(self, data, forecast_datetime, target_pollutant, prediction):
        """Create a new row for the forecast data."""
        # Get the latest row as template
        latest_row = data.iloc[-1].copy()
        
        # Update datetime and target pollutant
        new_row = pd.DataFrame([latest_row])
        new_row['datetime'] = forecast_datetime
        new_row[target_pollutant] = prediction
        
        # Update temporal features
        new_row['year'] = forecast_datetime.year
        new_row['month'] = forecast_datetime.month
        new_row['day'] = forecast_datetime.day
        new_row['hour'] = forecast_datetime.hour
        new_row['dayofweek'] = forecast_datetime.dayofweek
        
        # Update cyclical features
        new_row['hour_sin'] = np.sin(2 * np.pi * forecast_datetime.hour / 24)
        new_row['hour_cos'] = np.cos(2 * np.pi * forecast_datetime.hour / 24)
        new_row['month_sin'] = np.sin(2 * np.pi * forecast_datetime.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * forecast_datetime.month / 12)
        new_row['day_sin'] = np.sin(2 * np.pi * forecast_datetime.dayofweek / 7)
        new_row['day_cos'] = np.cos(2 * np.pi * forecast_datetime.dayofweek / 7)
        
        # Update season
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        new_row['season'] = season_map[forecast_datetime.month]
        
        return new_row
    
    def plot_forecast(self, historical_data, forecast_data, target_pollutant):
        """Plot historical data and forecast."""
        fig = go.Figure()
        
        # Plot historical data (last 7 days)
        recent_data = historical_data.tail(168)  # Last 7 days
        
        fig.add_trace(go.Scatter(
            x=recent_data['datetime'],
            y=recent_data[target_pollutant],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['datetime'],
            y=forecast_data['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add vertical line to separate historical and forecast
        last_historical_date = historical_data['datetime'].max()
        fig.add_vline(
            x=last_historical_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start"
        )
        
        fig.update_layout(
            title=f'{target_pollutant} Forecast',
            xaxis_title='Date',
            yaxis_title=f'{target_pollutant} Concentration (μg/m³)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def calculate_forecast_confidence(self, forecast_data, confidence_level=0.95):
        """Calculate confidence intervals for forecasts (simplified approach)."""
        # Simple approach: use increasing uncertainty with forecast horizon
        forecasts = forecast_data['forecast'].values
        steps = forecast_data['step'].values
        
        # Base uncertainty (could be improved with actual model uncertainty)
        base_uncertainty = np.std(forecasts) * 0.1
        
        # Increasing uncertainty with time
        uncertainties = base_uncertainty * (1 + 0.1 * steps)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        lower_bound = forecasts - z_score * uncertainties
        upper_bound = forecasts + z_score * uncertainties
        
        # Ensure non-negative values
        lower_bound = np.maximum(lower_bound, 0)
        
        forecast_data = forecast_data.copy()
        forecast_data['lower_bound'] = lower_bound
        forecast_data['upper_bound'] = upper_bound
        forecast_data['uncertainty'] = uncertainties
        
        return forecast_data
    
    def plot_forecast_with_confidence(self, historical_data, forecast_data, target_pollutant):
        """Plot forecast with confidence intervals."""
        # Calculate confidence intervals
        forecast_with_ci = self.calculate_forecast_confidence(forecast_data)
        
        fig = go.Figure()
        
        # Plot historical data
        recent_data = historical_data.tail(168)
        fig.add_trace(go.Scatter(
            x=recent_data['datetime'],
            y=recent_data[target_pollutant],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Plot confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_with_ci['datetime'],
            y=forecast_with_ci['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_with_ci['datetime'],
            y=forecast_with_ci['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_with_ci['datetime'],
            y=forecast_with_ci['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add vertical line
        last_historical_date = historical_data['datetime'].max()
        fig.add_vline(
            x=last_historical_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start"
        )
        
        fig.update_layout(
            title=f'{target_pollutant} Forecast with Confidence Interval',
            xaxis_title='Date',
            yaxis_title=f'{target_pollutant} Concentration (μg/m³)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def export_forecast(self, forecast_data, filename=None):
        """Export forecast data to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_{timestamp}.csv"
        
        forecast_data.to_csv(filename, index=False)
        return filename
