import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import streamlit as st

class Utils:
    """Utility functions for the AirSense application."""
    
    def __init__(self):
        self.config_dir = "config"
        self.exports_dir = "exports"
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def validate_data_format(self, data):
        """Validate that uploaded data has the correct format."""
        required_checks = {
            'has_datetime': False,
            'has_pollutants': False,
            'valid_data_types': False,
            'sufficient_data': False
        }
        
        errors = []
        warnings = []
        
        # Check for datetime column
        datetime_cols = ['datetime', 'date', 'time', 'timestamp']
        if any(col in data.columns.str.lower() for col in datetime_cols):
            required_checks['has_datetime'] = True
        else:
            errors.append("No datetime column found. Expected one of: datetime, date, time, timestamp")
        
        # Check for pollutant columns
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        found_pollutants = [col for col in pollutant_cols if col in data.columns or 
                           col.lower() in data.columns.str.lower()]
        
        if len(found_pollutants) > 0:
            required_checks['has_pollutants'] = True
        else:
            errors.append(f"No pollutant columns found. Expected at least one of: {', '.join(pollutant_cols)}")
        
        # Check data types
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                required_checks['valid_data_types'] = True
            else:
                warnings.append("No numeric columns found. Data may need preprocessing.")
        except Exception as e:
            warnings.append(f"Could not validate data types: {str(e)}")
        
        # Check data sufficiency
        if len(data) >= 24:  # At least 24 hours of data
            required_checks['sufficient_data'] = True
        else:
            warnings.append(f"Dataset has only {len(data)} records. Recommend at least 24 for meaningful analysis.")
        
        return {
            'is_valid': all(required_checks.values()),
            'checks': required_checks,
            'errors': errors,
            'warnings': warnings
        }
    
    def export_data(self, data, filename=None, format='csv'):
        """Export data to various formats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"airsense_data_{timestamp}"
        
        filepath = os.path.join(self.exports_dir, f"{filename}.{format}")
        
        try:
            if format.lower() == 'csv':
                data.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            elif format.lower() == 'excel':
                data.to_excel(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return filepath
        except Exception as e:
            raise Exception(f"Error exporting data: {str(e)}")
    
    def save_config(self, config_name, config_data):
        """Save configuration settings."""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            return config_path
        except Exception as e:
            raise Exception(f"Error saving config: {str(e)}")
    
    def load_config(self, config_name):
        """Load configuration settings."""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading config: {str(e)}")
    
    def calculate_data_quality_score(self, data):
        """Calculate a data quality score based on various factors."""
        score = 0
        max_score = 100
        factors = {}
        
        # Completeness (40 points)
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        completeness_score = max(0, 40 * (1 - missing_ratio))
        score += completeness_score
        factors['completeness'] = completeness_score
        
        # Consistency (20 points)
        # Check for reasonable pollutant values
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        consistency_issues = 0
        
        for col in pollutant_cols:
            if col in data.columns:
                # Check for negative values
                if (data[col] < 0).any():
                    consistency_issues += 1
                # Check for extremely high values (likely outliers)
                if col in ['PM2.5', 'PM10'] and (data[col] > 1000).any():
                    consistency_issues += 1
                elif col in ['NO2', 'SO2', 'O3'] and (data[col] > 500).any():
                    consistency_issues += 1
                elif col == 'CO' and (data[col] > 50).any():
                    consistency_issues += 1
        
        consistency_score = max(0, 20 * (1 - consistency_issues / max(1, len(pollutant_cols))))
        score += consistency_score
        factors['consistency'] = consistency_score
        
        # Temporal continuity (20 points)
        if 'datetime' in data.columns:
            try:
                data_sorted = data.sort_values('datetime')
                time_diffs = data_sorted['datetime'].diff().dt.total_seconds() / 3600  # hours
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 1
                
                # Check for gaps
                large_gaps = (time_diffs > expected_interval * 2).sum()
                temporal_score = max(0, 20 * (1 - large_gaps / len(data)))
                score += temporal_score
                factors['temporal_continuity'] = temporal_score
            except:
                factors['temporal_continuity'] = 0
        else:
            factors['temporal_continuity'] = 0
        
        # Data volume (20 points)
        # More data points generally mean better analysis
        if len(data) >= 8760:  # 1 year of hourly data
            volume_score = 20
        elif len(data) >= 720:  # 1 month of hourly data
            volume_score = 15
        elif len(data) >= 168:  # 1 week of hourly data
            volume_score = 10
        elif len(data) >= 24:   # 1 day of hourly data
            volume_score = 5
        else:
            volume_score = 0
        
        score += volume_score
        factors['data_volume'] = volume_score
        
        return {
            'total_score': min(score, max_score),
            'grade': self._get_quality_grade(score),
            'factors': factors,
            'recommendations': self._get_quality_recommendations(factors)
        }
    
    def _get_quality_grade(self, score):
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _get_quality_recommendations(self, factors):
        """Generate recommendations based on quality factors."""
        recommendations = []
        
        if factors['completeness'] < 30:
            recommendations.append("Address missing data through imputation or collection of additional data.")
        
        if factors['consistency'] < 15:
            recommendations.append("Review data for outliers and inconsistent values that may indicate measurement errors.")
        
        if factors['temporal_continuity'] < 15:
            recommendations.append("Fill temporal gaps in the data or ensure consistent measurement intervals.")
        
        if factors['data_volume'] < 10:
            recommendations.append("Collect more data points for robust analysis and modeling.")
        
        if not recommendations:
            recommendations.append("Data quality is good. Proceed with analysis and modeling.")
        
        return recommendations
    
    def format_pollutant_value(self, value, pollutant):
        """Format pollutant values with appropriate units and precision."""
        if pd.isna(value):
            return "N/A"
        
        # Different pollutants have different typical ranges and precision needs
        if pollutant == 'CO':
            return f"{value:.1f} mg/m³"
        elif pollutant in ['PM2.5', 'PM10']:
            return f"{value:.0f} μg/m³"
        else:
            return f"{value:.1f} μg/m³"
    
    def get_system_info(self):
        """Get system information for debugging."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'streamlit_version': st.__version__,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
        
        return info
    
    def log_activity(self, activity, details=None):
        """Log user activities for monitoring and debugging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'details': details or {}
        }
        
        log_file = os.path.join(self.config_dir, 'activity_log.json')
        
        # Load existing logs
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new log entry
        logs.append(log_entry)
        
        # Keep only last 100 entries
        logs = logs[-100:]
        
        # Save logs
        try:
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except:
            pass  # Fail silently if logging fails
    
    def clean_temp_files(self, older_than_hours=24):
        """Clean temporary files older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for directory in [self.exports_dir, self.config_dir]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time < cutoff_time:
                            os.remove(filepath)
                    except:
                        continue  # Skip files that can't be processed
    
    def create_download_link(self, data, filename, format='csv'):
        """Create a download link for data export."""
        try:
            if format.lower() == 'csv':
                csv_data = data.to_csv(index=False)
                return st.download_button(
                    label=f"Download {filename}.csv",
                    data=csv_data,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            elif format.lower() == 'json':
                json_data = data.to_json(orient='records', date_format='iso')
                return st.download_button(
                    label=f"Download {filename}.json",
                    data=json_data,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error creating download link: {str(e)}")
            return None
