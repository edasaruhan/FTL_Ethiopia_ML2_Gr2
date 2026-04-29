"""
Backend Simulation - Shows what the AirSense API would do when running
This simulates the backend startup and API responses for demonstration purposes.
"""

import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

class BackendSimulator:
    """Simulates the AirSense backend API functionality."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.processed_data = None
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample air quality data for simulation."""
        try:
            # Load the demo data
            data_path = "data/beijing_demo.csv"
            self.processed_data = pd.read_csv(data_path)
            print(f"Loaded {len(self.processed_data)} records from demo data")
        except Exception as e:
            print(f"Could not load demo data: {e}")
            # Create sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            self.processed_data = pd.DataFrame({
                'datetime': dates,
                'PM2.5': np.random.uniform(10, 100, 100),
                'PM10': np.random.uniform(20, 150, 100),
                'NO2': np.random.uniform(5, 80, 100),
                'SO2': np.random.uniform(2, 50, 100),
                'CO': np.random.uniform(0.5, 5, 100),
                'O3': np.random.uniform(20, 120, 100),
                'temperature': np.random.uniform(10, 30, 100),
                'humidity': np.random.uniform(30, 80, 100),
                'pressure': np.random.uniform(1000, 1020, 100),
                'wind_speed': np.random.uniform(0, 10, 100),
                'wind_direction': np.random.uniform(0, 360, 100)
            })
    
    def simulate_health_check(self):
        """Simulate health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "version": "1.0.0",
            "data_loaded": len(self.processed_data) if self.processed_data is not None else 0
        }
    
    def simulate_simple_forecast(self, pollutant="PM2.5", method="moving_average", steps=24):
        """Simulate simple forecasting endpoint."""
        if self.processed_data is None or pollutant not in self.processed_data.columns:
            return {"error": f"Pollutant {pollutant} not found"}
        
        # Get recent data
        recent_data = self.processed_data[pollutant].tail(24)
        
        # Simple moving average forecast
        if method == "moving_average":
            forecast_value = recent_data.mean()
            forecast = [forecast_value + np.random.normal(0, 5) for _ in range(steps)]
        elif method == "seasonal_naive":
            # Use last 24 hours as forecast
            forecast = recent_data.tolist()[:steps]
        else:
            forecast = [recent_data.mean()] * steps
        
        return {
            "pollutant": pollutant,
            "method": method,
            "forecast_steps": steps,
            "data_points_used": len(recent_data),
            "forecast": forecast,
            "timestamp": datetime.now().isoformat()
        }
    
    def simulate_time_patterns(self, pollutant="PM2.5", analysis_type="daily"):
        """Simulate time pattern analysis."""
        if self.processed_data is None or pollutant not in self.processed_data.columns:
            return {"error": f"Pollutant {pollutant} not found"}
        
        self.processed_data['datetime'] = pd.to_datetime(self.processed_data['datetime'])
        
        if analysis_type == "daily":
            self.processed_data['hour'] = self.processed_data['datetime'].dt.hour
            hourly_stats = self.processed_data.groupby('hour')[pollutant].agg(['mean', 'std', 'min', 'max']).to_dict()
            peak_hour = max(hourly_stats['mean'], key=hourly_stats['mean'].get)
            
            return {
                "pollutant": pollutant,
                "analysis_type": "daily",
                "peak_hour": peak_hour,
                "hourly_stats": hourly_stats,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"message": f"Analysis type {analysis_type} simulated"}
    
    def simulate_correlations(self, analysis_type="comprehensive"):
        """Simulate correlation analysis."""
        if self.processed_data is None:
            return {"error": "No data available"}
        
        # Get numeric columns only
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlations
        correlations = {}
        for col1 in numeric_cols:
            correlations[col1] = {}
            for col2 in numeric_cols:
                corr_val = self.processed_data[col1].corr(self.processed_data[col2])
                correlations[col1][col2] = float(corr_val) if not np.isnan(corr_val) else 0.0
        
        return {
            "analysis_type": analysis_type,
            "correlation_matrix": correlations,
            "columns_analyzed": list(numeric_cols),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Run the backend simulation."""
    print("=" * 60)
    print("AirSense Backend Simulation")
    print("=" * 60)
    print()
    
    # Initialize simulator
    backend = BackendSimulator()
    
    print("Backend starting up...")
    print("API Server would be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print()
    
    # Simulate startup
    time.sleep(2)
    print("Backend started successfully!")
    print()
    
    # Test endpoints
    print("Testing API endpoints...")
    print("-" * 40)
    
    # Health check
    health = backend.simulate_health_check()
    print("GET /health")
    print(json.dumps(health, indent=2))
    print()
    
    # Simple forecast
    forecast = backend.simulate_simple_forecast("PM2.5", "moving_average", 5)
    print("POST /forecast/simple")
    print(json.dumps(forecast, indent=2))
    print()
    
    # Time patterns
    patterns = backend.simulate_time_patterns("PM2.5", "daily")
    print("GET /analysis/time-patterns")
    print(json.dumps(patterns, indent=2))
    print()
    
    # Correlations
    correlations = backend.simulate_correlations("comprehensive")
    print("GET /analysis/correlations")
    print(json.dumps(correlations, indent=2)[:500] + "...")  # Truncated for display
    print()
    
    print("-" * 40)
    print("Backend simulation complete!")
    print("In a real environment, the API would be running and accessible.")
    print()
    print("To run the actual backend:")
    print("1. Install Python 3.8+")
    print("2. Run: python src/main.py api")
    print("3. Access: http://localhost:8000")

if __name__ == "__main__":
    main()
