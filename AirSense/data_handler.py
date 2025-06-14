import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os

class DataHandler:
    """Handles data loading, preprocessing, and validation for air quality data."""
    
    def __init__(self):
        self.required_columns = ['datetime', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        self.optional_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
    
    def load_uploaded_data(self, uploaded_file):
        """Load and validate uploaded CSV data."""
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file)
            
            # Validate and preprocess data
            data = self._preprocess_data(data)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading uploaded data: {str(e)}")
    
    def load_demo_data(self):
        """Load Beijing demo dataset."""
        try:
            # Check if demo data exists
            demo_path = "data/beijing_demo.csv"
            if os.path.exists(demo_path):
                data = pd.read_csv(demo_path)
            else:
                # Generate synthetic Beijing-like data for demo
                data = self._generate_demo_data()
            
            # Preprocess data
            data = self._preprocess_data(data)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading demo data: {str(e)}")
    
    def _preprocess_data(self, data):
        """Preprocess and validate air quality data."""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Standardize column names
        data.columns = data.columns.str.strip().str.lower()
        
        # Handle datetime column
        datetime_cols = ['datetime', 'date', 'time', 'timestamp']
        datetime_col = None
        
        for col in datetime_cols:
            if col in data.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            # If no datetime column, create one
            if len(data) > 0:
                start_date = datetime.now() - timedelta(days=len(data))
                data['datetime'] = pd.date_range(start=start_date, periods=len(data), freq='h')
            else:
                raise ValueError("No datetime column found and cannot create one for empty dataset")
        else:
            # Convert to datetime
            data['datetime'] = pd.to_datetime(data[datetime_col])
            if datetime_col != 'datetime':
                data = data.drop(columns=[datetime_col])
        
        # Ensure required pollutant columns exist
        pollutant_mapping = {
            'pm2.5': 'PM2.5',
            'pm25': 'PM2.5',
            'pm_2.5': 'PM2.5',
            'pm10': 'PM10',
            'pm_10': 'PM10',
            'no2': 'NO2',
            'so2': 'SO2',
            'co': 'CO',
            'o3': 'O3'
        }
        
        # Rename columns to standard names
        for old_name, new_name in pollutant_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # Add missing pollutant columns with NaN values
        for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
            if col not in data.columns:
                data[col] = np.nan
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Add temporal features
        data = self._add_temporal_features(data)
        
        # Sort by datetime
        data = data.sort_values('datetime').reset_index(drop=True)
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset."""
        # For pollutant columns, use forward fill then backward fill
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        
        for col in pollutant_cols:
            if col in data.columns:
                # Remove extreme outliers (values > 99.9th percentile)
                if data[col].notna().sum() > 0:
                    upper_bound = data[col].quantile(0.999)
                    data.loc[data[col] > upper_bound, col] = np.nan
                
                # Fill missing values
                data[col] = data[col].ffill().bfill()
                
                # If still missing, fill with median
                if data[col].isna().sum() > 0:
                    median_val = data[col].median()
                    if not np.isnan(median_val):
                        data[col] = data[col].fillna(median_val)
                    else:
                        # If median is also NaN, fill with a default value
                        default_values = {
                            'PM2.5': 25.0, 'PM10': 50.0, 'NO2': 30.0,
                            'SO2': 20.0, 'CO': 1.0, 'O3': 80.0
                        }
                        data[col] = data[col].fillna(default_values.get(col, 0))
        
        return data
    
    def _add_temporal_features(self, data):
        """Add temporal features for better model performance."""
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['day'] = data['datetime'].dt.day
        data['hour'] = data['datetime'].dt.hour
        data['dayofweek'] = data['datetime'].dt.dayofweek
        data['season'] = data['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Cyclical encoding for temporal features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
        
        return data
    
    def _generate_demo_data(self):
        """Generate synthetic Beijing-like air quality data for demonstration."""
        # Generate 30 days of hourly data
        n_hours = 30 * 24
        start_date = datetime.now() - timedelta(days=30)
        
        # Create datetime index
        dates = pd.date_range(start=start_date, periods=n_hours, freq='H')
        
        # Generate realistic Beijing air quality data patterns
        np.random.seed(42)  # For reproducibility
        
        # Base patterns with seasonal and diurnal variations
        hour_of_day = dates.hour.values
        day_of_year = dates.dayofyear.values
        
        # PM2.5 with higher values in winter and rush hours
        pm25_base = 45 + 20 * np.sin(2 * np.pi * day_of_year / 365) + \
                   15 * np.sin(2 * np.pi * hour_of_day / 24) + \
                   np.random.normal(0, 15, n_hours)
        pm25_base = np.maximum(pm25_base, 5)  # Minimum value
        
        # PM10 correlated with PM2.5 but higher
        pm10_base = pm25_base * 1.8 + np.random.normal(0, 10, n_hours)
        pm10_base = np.maximum(pm10_base, 10)
        
        # NO2 with traffic patterns
        no2_base = 35 + 15 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/4) + \
                  np.random.normal(0, 8, n_hours)
        no2_base = np.maximum(no2_base, 5)
        
        # SO2 with industrial patterns
        so2_base = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + \
                  np.random.normal(0, 5, n_hours)
        so2_base = np.maximum(so2_base, 2)
        
        # CO with traffic patterns
        co_base = 1.2 + 0.8 * np.sin(2 * np.pi * hour_of_day / 24) + \
                 np.random.normal(0, 0.3, n_hours)
        co_base = np.maximum(co_base, 0.1)
        
        # O3 with photochemical patterns (higher during day)
        o3_base = 80 + 40 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2) + \
                 np.random.normal(0, 15, n_hours)
        o3_base = np.maximum(o3_base, 10)
        
        # Create DataFrame
        data = pd.DataFrame({
            'datetime': dates,
            'PM2.5': pm25_base,
            'PM10': pm10_base,
            'NO2': no2_base,
            'SO2': so2_base,
            'CO': co_base,
            'O3': o3_base,
            'temperature': 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 
                          5 * np.sin(2 * np.pi * hour_of_day / 24) + 
                          np.random.normal(0, 3, n_hours),
            'humidity': 60 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) + 
                       np.random.normal(0, 10, n_hours),
            'pressure': 1013 + np.random.normal(0, 8, n_hours),
            'wind_speed': np.maximum(np.random.exponential(3, n_hours), 0.1),
            'wind_direction': np.random.uniform(0, 360, n_hours)
        })
        
        return data
    
    def get_data_summary(self, data):
        """Get summary statistics of the data."""
        summary = {
            'shape': data.shape,
            'date_range': (data['datetime'].min(), data['datetime'].max()),
            'missing_values': data.isnull().sum().to_dict(),
            'pollutant_stats': {}
        }
        
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for col in pollutant_cols:
            if col in data.columns:
                summary['pollutant_stats'][col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
        
        return summary
