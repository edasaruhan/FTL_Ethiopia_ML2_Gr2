import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class TimeSeriesForecaster:
    """Advanced time series forecasting with ARIMA and LSTM models."""
    
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
        # Create model directory
        import os
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_data(self, df, target_col, datetime_col='datetime'):
        """Prepare time series data for modeling."""
        # Ensure datetime is index
        data = df.copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data = data.set_index(datetime_col)
        data = data.sort_index()
        
        # Handle missing values
        data[target_col] = data[target_col].ffill().bfill()
        
        return data
    
    def decompose_series(self, series, period=24):
        """Decompose time series into trend, seasonal, and residual components."""
        try:
            decomposition = seasonal_decompose(series, period=period, model='additive')
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except Exception as e:
            self.logger.warning(f"Decomposition failed: {e}")
            return None
    
    def train_arima(self, data, target_col, order=(1,1,1), seasonal_order=(1,1,1,24)):
        """Train ARIMA model for time series forecasting."""
        try:
            series = data[target_col]
            
            # Fit ARIMA model
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            # Store model
            model_name = f"arima_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_name] = fitted_model
            
            # Save model
            model_path = f"{self.model_dir}/{model_name}.pkl"
            joblib.dump(fitted_model, model_path)
            
            self.logger.info(f"ARIMA model trained and saved: {model_path}")
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'fitted_model': fitted_model
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            return None
    
    def train_lstm(self, data, target_col, sequence_length=24, epochs=50, batch_size=32):
        """Train LSTM model for time series forecasting."""
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available for LSTM training")
            return None
        
        try:
            # Prepare data
            series = data[target_col].values.reshape(-1, 1)
            
            # Scale data
            scaler = StandardScaler()
            scaled_series = scaler.fit_transform(series)
            
            # Create sequences
            X, y = self._create_sequences(scaled_series, sequence_length)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Store model and scaler
            model_name = f"lstm_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save model and scaler
            model_path = f"{self.model_dir}/{model_name}.h5"
            scaler_path = f"{self.model_dir}/{model_name}_scaler.pkl"
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            self.logger.info(f"LSTM model trained and saved: {model_path}")
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'history': history.history,
                'model': model,
                'scaler': scaler
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            return None
    
    def _create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def forecast_arima(self, model_name, steps=24):
        """Generate forecasts using trained ARIMA model."""
        try:
            model = self.models.get(model_name)
            if model is None:
                # Try to load from disk
                model_path = f"{self.model_dir}/{model_name}.pkl"
                model = joblib.load(model_path)
                self.models[model_name] = model
            
            # Generate forecast
            forecast = model.forecast(steps=steps)
            conf_int = model.get_forecast(steps=steps).conf_int()
            
            return {
                'forecast': forecast,
                'confidence_interval': conf_int,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA forecasting failed: {e}")
            return None
    
    def forecast_lstm(self, model_name, data, target_col, steps=24, sequence_length=24):
        """Generate forecasts using trained LSTM model."""
        if not TF_AVAILABLE:
            return None
        
        try:
            model = self.models.get(model_name)
            scaler = self.scalers.get(model_name)
            
            if model is None:
                # Try to load from disk
                model_path = f"{self.model_dir}/{model_name}.h5"
                scaler_path = f"{self.model_dir}/{model_name}_scaler.pkl"
                
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
            
            # Prepare last sequence
            series = data[target_col].values.reshape(-1, 1)
            scaled_series = scaler.transform(series)
            
            last_sequence = scaled_series[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Generate forecasts
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                pred = model.predict(current_sequence, verbose=0)
                forecasts.append(pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]
            
            # Inverse transform
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = scaler.inverse_transform(forecasts).flatten()
            
            return {
                'forecast': forecasts,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"LSTM forecasting failed: {e}")
            return None
    
    def evaluate_model(self, model_name, test_data, target_col):
        """Evaluate model performance on test data."""
        try:
            if 'arima' in model_name:
                return self._evaluate_arima(model_name, test_data, target_col)
            elif 'lstm' in model_name:
                return self._evaluate_lstm(model_name, test_data, target_col)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return None
    
    def _evaluate_arima(self, model_name, test_data, target_col):
        """Evaluate ARIMA model."""
        model = self.models.get(model_name)
        if model is None:
            model = joblib.load(f"{self.model_dir}/{model_name}.pkl")
        
        # Generate predictions
        predictions = model.forecast(steps=len(test_data))
        actual = test_data[target_col].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actual': actual
        }
    
    def _evaluate_lstm(self, model_name, test_data, target_col, sequence_length=24):
        """Evaluate LSTM model."""
        if not TF_AVAILABLE:
            return None
        
        model = self.models.get(model_name)
        scaler = self.scalers.get(model_name)
        
        if model is None:
            from tensorflow.keras.models import load_model
            model = load_model(f"{self.model_dir}/{model_name}.h5")
            scaler = joblib.load(f"{self.model_dir}/{model_name}_scaler.pkl")
        
        # Prepare test data
        series = test_data[target_col].values.reshape(-1, 1)
        scaled_series = scaler.transform(series)
        
        X_test, y_test = self._create_sequences(scaled_series, sequence_length)
        
        # Generate predictions
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions).flatten()
        actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actual': actual
        }
