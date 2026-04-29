"""Advanced time series forecasting models."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..core.config import get_settings
from ..core.logging import get_logger, log_performance
from ..core.exceptions import ModelError, ForecastError

# Try to import TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class BaseModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self, name: str):
        self.name = name
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.model = None
        self.is_trained = False
        self.metadata = {}
    
    @abstractmethod
    def train(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """Generate forecasts."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model."""
        pass
    
    def evaluate(self, test_data: pd.Series, **kwargs) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ModelError(f"Model {self.name} is not trained")
        
        try:
            # Generate forecasts for test data length
            forecasts = self.forecast(len(test_data), **kwargs)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecasts)
            mse = mean_squared_error(test_data, forecasts)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_data, forecasts)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
            
            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape)
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise ModelError(f"Model evaluation failed: {e}")


class ARIMAModel(BaseModel):
    """ARIMA time series model with automatic parameter selection."""
    
    def __init__(self):
        super().__init__("ARIMA")
        self.auto_params = True
        self.seasonal_period = 24  # Hourly data
    
    @log_performance
    def train(self, data: pd.Series, order: Optional[Tuple[int, int, int]] = None,
              seasonal_order: Optional[Tuple[int, int, int, int]] = None,
              auto_params: bool = True) -> Dict[str, Any]:
        """Train ARIMA model."""
        try:
            self.logger.info("Training ARIMA model", data_points=len(data))
            
            # Handle missing values
            data = data.ffill().bfill()
            
            if len(data) < 100:
                raise ModelError("Insufficient data for ARIMA training (minimum 100 points)")
            
            # Automatic parameter selection
            if auto_params and (order is None or seasonal_order is None):
                order, seasonal_order = self._auto_select_params(data)
            
            # Default parameters if auto-selection fails
            if order is None:
                order = (1, 1, 1)
            if seasonal_order is None:
                seasonal_order = (1, 1, 1, self.seasonal_period)
            
            # Fit model
            self.model = ARIMA(data, order=order, seasonal_order=seasonal_order)
            self.model = self.model.fit()
            
            # Store metadata
            self.metadata = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": self.model.aic,
                "bic": self.model.bic,
                "training_samples": len(data),
                "training_date": datetime.now().isoformat()
            }
            
            self.is_trained = True
            
            self.logger.info("ARIMA model trained successfully", **self.metadata)
            return self.metadata
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            raise ModelError(f"ARIMA training failed: {e}")
    
    def _auto_select_params(self, data: pd.Series) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Automatically select ARIMA parameters using grid search."""
        try:
            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_seasonal = (1, 1, 1, self.seasonal_period)
            
            # Simplified parameter search for performance
            p_values = [0, 1, 2]
            d_values = [0, 1]
            q_values = [0, 1, 2]
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Try seasonal parameters
            P_values = [0, 1]
            D_values = [0, 1]
            Q_values = [0, 1]
            
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        try:
                            model = ARIMA(data, order=best_order, 
                                        seasonal_order=(P, D, Q, self.seasonal_period))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_seasonal = (P, D, Q, self.seasonal_period)
                        except:
                            continue
            
            self.logger.info("Auto-selected parameters", 
                           order=best_order, seasonal_order=best_seasonal, aic=best_aic)
            
            return best_order, best_seasonal
            
        except Exception as e:
            self.logger.warning(f"Auto parameter selection failed: {e}, using defaults")
            return (1, 1, 1), (1, 1, 1, self.seasonal_period)
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> np.ndarray:
        """Generate forecasts with confidence intervals."""
        if not self.is_trained:
            raise ModelError("ARIMA model is not trained")
        
        try:
            # Generate forecast
            forecast_result = self.model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)
            
            self.logger.info("ARIMA forecast generated", steps=steps)
            return forecast.values
            
        except Exception as e:
            self.logger.error(f"ARIMA forecasting failed: {e}")
            raise ForecastError(f"ARIMA forecasting failed: {e}")
    
    def forecast_with_confidence(self, steps: int, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals."""
        if not self.is_trained:
            raise ModelError("ARIMA model is not trained")
        
        try:
            forecast_result = self.model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)
            
            return {
                "forecast": forecast.values,
                "confidence_lower": conf_int.iloc[:, 0].values,
                "confidence_upper": conf_int.iloc[:, 1].values
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA forecasting with confidence failed: {e}")
            raise ForecastError(f"ARIMA forecasting failed: {e}")
    
    def save(self, path: str) -> None:
        """Save ARIMA model."""
        if not self.is_trained:
            raise ModelError("Cannot save untrained model")
        
        try:
            model_data = {
                "model": self.model,
                "metadata": self.metadata,
                "name": self.name
            }
            joblib.dump(model_data, path)
            self.logger.info("ARIMA model saved", path=path)
            
        except Exception as e:
            self.logger.error(f"Failed to save ARIMA model: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def load(self, path: str) -> None:
        """Load ARIMA model."""
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.metadata = model_data["metadata"]
            self.name = model_data["name"]
            self.is_trained = True
            self.logger.info("ARIMA model loaded", path=path)
            
        except Exception as e:
            self.logger.error(f"Failed to load ARIMA model: {e}")
            raise ModelError(f"Failed to load model: {e}")


class LSTMModel(BaseModel):
    """LSTM neural network for time series forecasting."""
    
    def __init__(self, sequence_length: int = 24, lstm_units: int = 50):
        super().__init__("LSTM")
        
        if not TF_AVAILABLE:
            raise ModelError("TensorFlow is not available for LSTM model")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.scaler = StandardScaler()
        self.history = None
    
    @log_performance
    def train(self, data: pd.Series, epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train LSTM model."""
        try:
            self.logger.info("Training LSTM model", data_points=len(data))
            
            # Handle missing values
            data = data.ffill().bfill()
            
            if len(data) < self.sequence_length + 50:
                raise ModelError(f"Insufficient data for LSTM training (minimum {self.sequence_length + 50} points)")
            
            # Prepare data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            X, y = self._create_sequences(scaled_data)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self._build_model()
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Store metadata
            self.metadata = {
                "sequence_length": self.sequence_length,
                "lstm_units": self.lstm_units,
                "epochs": epochs,
                "batch_size": batch_size,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "training_date": datetime.now().isoformat(),
                "final_loss": float(self.history.history['loss'][-1]),
                "final_val_loss": float(self.history.history['val_loss'][-1])
            }
            
            self.is_trained = True
            
            self.logger.info("LSTM model trained successfully", **self.metadata)
            return self.metadata
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            raise ModelError(f"LSTM training failed: {e}")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """Generate multi-step forecasts."""
        if not self.is_trained:
            raise ModelError("LSTM model is not trained")
        
        try:
            # Get the last sequence from training data
            # Note: In practice, you'd want to store the last sequence during training
            # For now, we'll generate forecasts recursively
            forecasts = []
            
            # Start with the last known sequence (simplified approach)
            # In production, you'd maintain the last sequence
            current_sequence = np.random.randn(1, self.sequence_length, 1)  # Placeholder
            
            for _ in range(steps):
                pred = self.model.predict(current_sequence, verbose=0)
                forecasts.append(pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]
            
            # Inverse transform
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = self.scaler.inverse_transform(forecasts).flatten()
            
            self.logger.info("LSTM forecast generated", steps=steps)
            return forecasts
            
        except Exception as e:
            self.logger.error(f"LSTM forecasting failed: {e}")
            raise ForecastError(f"LSTM forecasting failed: {e}")
    
    def save(self, path: str) -> None:
        """Save LSTM model."""
        if not self.is_trained:
            raise ModelError("Cannot save untrained model")
        
        try:
            # Save model architecture and weights
            model_path = path.replace('.pkl', '.h5')
            self.model.save(model_path)
            
            # Save scaler and metadata
            additional_data = {
                "scaler": self.scaler,
                "metadata": self.metadata,
                "name": self.name,
                "sequence_length": self.sequence_length
            }
            joblib.dump(additional_data, path)
            
            self.logger.info("LSTM model saved", path=path)
            
        except Exception as e:
            self.logger.error(f"Failed to save LSTM model: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def load(self, path: str) -> None:
        """Load LSTM model."""
        try:
            # Load additional data
            additional_data = joblib.load(path)
            self.scaler = additional_data["scaler"]
            self.metadata = additional_data["metadata"]
            self.name = additional_data["name"]
            self.sequence_length = additional_data["sequence_length"]
            
            # Load model
            model_path = path.replace('.pkl', '.h5')
            self.model = load_model(model_path)
            self.is_trained = True
            
            self.logger.info("LSTM model loaded", path=path)
            
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            raise ModelError(f"Failed to load model: {e}")


class TimeSeriesForecaster:
    """Main forecaster class that manages multiple models."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.models = {}
        self.model_registry = {}
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a model to the forecaster."""
        self.models[name] = model
        self.logger.info("Model added to forecaster", model_name=name, model_type=model.name)
    
    def train_model(self, model_type: str, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train a specific model type."""
        try:
            if model_type.lower() == "arima":
                model = ARIMAModel()
            elif model_type.lower() == "lstm":
                model = LSTMModel()
            else:
                raise ModelError(f"Unsupported model type: {model_type}")
            
            # Train model
            metadata = model.train(data, **kwargs)
            
            # Store model
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.add_model(model_name, model)
            
            return {
                "model_name": model_name,
                "model_type": model_type,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelError(f"Model training failed: {e}")
    
    def forecast(self, model_name: str, steps: int, **kwargs) -> np.ndarray:
        """Generate forecasts using a specific model."""
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not found")
        
        return self.models[model_name].forecast(steps, **kwargs)
    
    def evaluate_model(self, model_name: str, test_data: pd.Series, **kwargs) -> Dict[str, float]:
        """Evaluate a specific model."""
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not found")
        
        return self.models[model_name].evaluate(test_data, **kwargs)
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save a specific model."""
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not found")
        
        self.models[model_name].save(path)
    
    def load_model(self, path: str, model_name: str) -> None:
        """Load a model from file."""
        # Determine model type from file
        if path.endswith('.h5') or 'lstm' in path.lower():
            model = LSTMModel()
        else:
            model = ARIMAModel()
        
        model.load(path)
        self.add_model(model_name, model)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models."""
        return {
            name: {
                "type": model.name,
                "trained": model.is_trained,
                "metadata": model.metadata
            }
            for name, model in self.models.items()
        }
