"""Simple forecasting methods for air quality analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..core.config import get_settings
from ..core.logging import get_logger, log_performance
from ..core.exceptions import ModelError, ForecastError


class SimpleForecaster:
    """Simple forecasting methods including moving averages and seasonal patterns."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.fitted_models = {}
    
    @log_performance
    def moving_average_forecast(self, 
                              data: pd.Series, 
                              window_size: int = 24,
                              forecast_steps: int = 24) -> Dict[str, Any]:
        """Generate moving average forecast."""
        try:
            self.logger.info("Computing moving average forecast", 
                           window_size=window_size, 
                           forecast_steps=forecast_steps)
            
            if len(data) < window_size:
                raise ModelError(f"Insufficient data for moving average (need at least {window_size} points)")
            
            # Calculate moving averages
            moving_avg = data.rolling(window=window_size).mean()
            
            # Simple forecast: use the last moving average value
            last_ma = moving_avg.iloc[-1]
            forecast = [last_ma] * forecast_steps
            
            # Calculate confidence intervals based on historical variance
            recent_data = data.tail(window_size * 2)
            std_error = recent_data.std()
            confidence_lower = [last_ma - 1.96 * std_error] * forecast_steps
            confidence_upper = [last_ma + 1.96 * std_error] * forecast_steps
            
            result = {
                "forecast": forecast,
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
                "method": "moving_average",
                "window_size": window_size,
                "last_moving_average": float(last_ma),
                "std_error": float(std_error),
                "data_points_used": len(data)
            }
            
            self.logger.info("Moving average forecast completed", 
                           forecast_len=len(forecast))
            return result
            
        except Exception as e:
            self.logger.error("Moving average forecast failed", error=str(e))
            raise ForecastError(f"Moving average forecast failed: {e}")
    
    @log_performance
    def weighted_moving_average_forecast(self,
                                       data: pd.Series,
                                       window_size: int = 24,
                                       forecast_steps: int = 24,
                                       weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Generate weighted moving average forecast."""
        try:
            self.logger.info("Computing weighted moving average forecast")
            
            if len(data) < window_size:
                raise ModelError(f"Insufficient data for weighted moving average")
            
            # Default weights (more recent data gets higher weight)
            if weights is None:
                weights = list(range(1, window_size + 1))
                weights = np.array(weights) / sum(weights)
            
            # Calculate weighted moving average
            forecast = []
            for _ in range(forecast_steps):
                recent_data = data.tail(window_size)
                weighted_avg = np.average(recent_data, weights=weights)
                forecast.append(weighted_avg)
                
                # For next step, use the forecasted value
                data = pd.concat([data, pd.Series([weighted_avg])])
            
            result = {
                "forecast": forecast,
                "method": "weighted_moving_average",
                "window_size": window_size,
                "weights": weights,
                "data_points_used": len(data) - forecast_steps
            }
            
            self.logger.info("Weighted moving average forecast completed")
            return result
            
        except Exception as e:
            self.logger.error("Weighted moving average forecast failed", error=str(e))
            raise ForecastError(f"Weighted moving average forecast failed: {e}")
    
    @log_performance
    def seasonal_naive_forecast(self,
                              data: pd.Series,
                              seasonal_period: int = 24,
                              forecast_steps: int = 24) -> Dict[str, Any]:
        """Generate seasonal naive forecast (use last season's values)."""
        try:
            self.logger.info("Computing seasonal naive forecast",
                           seasonal_period=seasonal_period)
            
            if len(data) < seasonal_period:
                raise ModelError(f"Insufficient data for seasonal naive (need at least {seasonal_period} points)")
            
            # Get the last seasonal period
            last_season = data.tail(seasonal_period).values
            
            # Forecast by repeating the last season
            forecast = []
            for i in range(forecast_steps):
                forecast.append(last_season[i % seasonal_period])
            
            result = {
                "forecast": forecast,
                "method": "seasonal_naive",
                "seasonal_period": seasonal_period,
                "last_season_values": last_season.tolist(),
                "data_points_used": len(data)
            }
            
            self.logger.info("Seasonal naive forecast completed")
            return result
            
        except Exception as e:
            self.logger.error("Seasonal naive forecast failed", error=str(e))
            raise ForecastError(f"Seasonal naive forecast failed: {e}")
    
    @log_performance
    def exponential_smoothing_forecast(self,
                                     data: pd.Series,
                                     alpha: float = 0.3,
                                     forecast_steps: int = 24) -> Dict[str, Any]:
        """Generate simple exponential smoothing forecast."""
        try:
            self.logger.info("Computing exponential smoothing forecast", alpha=alpha)
            
            if len(data) < 10:
                raise ModelError("Insufficient data for exponential smoothing")
            
            # Simple exponential smoothing
            smoothed = [data.iloc[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
            
            # Forecast using the last smoothed value
            last_smoothed = smoothed[-1]
            forecast = [last_smoothed] * forecast_steps
            
            result = {
                "forecast": forecast,
                "method": "exponential_smoothing",
                "alpha": alpha,
                "last_smoothed_value": float(last_smoothed),
                "data_points_used": len(data)
            }
            
            self.logger.info("Exponential smoothing forecast completed")
            return result
            
        except Exception as e:
            self.logger.error("Exponential smoothing forecast failed", error=str(e))
            raise ForecastError(f"Exponential smoothing forecast failed: {e}")
    
    def ensemble_forecast(self,
                         data: pd.Series,
                         forecast_steps: int = 24) -> Dict[str, Any]:
        """Generate ensemble forecast combining multiple simple methods."""
        try:
            self.logger.info("Computing ensemble forecast")
            
            # Generate forecasts from different methods
            methods = []
            forecasts = []
            
            # Moving average
            ma_result = self.moving_average_forecast(data, window_size=24, forecast_steps=forecast_steps)
            methods.append("moving_average")
            forecasts.append(ma_result["forecast"])
            
            # Weighted moving average
            wma_result = self.weighted_moving_average_forecast(data, window_size=24, forecast_steps=forecast_steps)
            methods.append("weighted_moving_average")
            forecasts.append(wma_result["forecast"])
            
            # Seasonal naive
            sn_result = self.seasonal_naive_forecast(data, seasonal_period=24, forecast_steps=forecast_steps)
            methods.append("seasonal_naive")
            forecasts.append(sn_result["forecast"])
            
            # Exponential smoothing
            es_result = self.exponential_smoothing_forecast(data, alpha=0.3, forecast_steps=forecast_steps)
            methods.append("exponential_smoothing")
            forecasts.append(es_result["forecast"])
            
            # Simple ensemble (average of all methods)
            ensemble_forecast = np.mean(forecasts, axis=0).tolist()
            
            # Calculate ensemble variance for confidence intervals
            forecast_variance = np.var(forecasts, axis=0)
            confidence_lower = ensemble_forecast - 1.96 * np.sqrt(forecast_variance)
            confidence_upper = ensemble_forecast + 1.96 * np.sqrt(forecast_variance)
            
            result = {
                "forecast": ensemble_forecast,
                "confidence_lower": confidence_lower.tolist(),
                "confidence_upper": confidence_upper.tolist(),
                "method": "ensemble",
                "individual_methods": methods,
                "individual_forecasts": forecasts,
                "ensemble_variance": forecast_variance.tolist(),
                "data_points_used": len(data)
            }
            
            self.logger.info("Ensemble forecast completed")
            return result
            
        except Exception as e:
            self.logger.error("Ensemble forecast failed", error=str(e))
            raise ForecastError(f"Ensemble forecast failed: {e}")
    
    def evaluate_simple_forecast(self,
                               actual: pd.Series,
                               forecast: List[float],
                               method_name: str = "unknown") -> Dict[str, float]:
        """Evaluate forecast accuracy."""
        try:
            if len(actual) != len(forecast):
                raise ModelError("Actual and forecast lengths must match")
            
            actual_values = actual.values
            forecast_values = np.array(forecast)
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_values - forecast_values))
            mse = np.mean((actual_values - forecast_values) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
            
            # Directional accuracy (for time series)
            actual_direction = np.diff(actual_values) > 0
            forecast_direction = np.diff(forecast_values) > 0
            directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
            
            result = {
                "method": method_name,
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(directional_accuracy),
                "forecast_length": len(forecast)
            }
            
            self.logger.info("Forecast evaluation completed", method=method_name, rmse=float(rmse))
            return result
            
        except Exception as e:
            self.logger.error("Forecast evaluation failed", error=str(e))
            raise ModelError(f"Forecast evaluation failed: {e}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available forecasting methods."""
        return [
            "moving_average",
            "weighted_moving_average", 
            "seasonal_naive",
            "exponential_smoothing",
            "ensemble"
        ]
