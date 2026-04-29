"""Tests for ML models module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.models.time_series import ARIMAModel, LSTMModel, TimeSeriesForecaster
from src.core.exceptions import ModelError, ForecastError


class TestARIMAModel:
    """Test cases for ARIMA model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        values = np.random.normal(50, 15, 200) + np.sin(np.arange(200) * 0.1) * 20
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def arima_model(self):
        """Create ARIMA model instance."""
        return ARIMAModel()
    
    def test_initialization(self, arima_model):
        """Test model initialization."""
        assert arima_model.name == "ARIMA"
        assert arima_model.seasonal_period == 24
        assert not arima_model.is_trained
    
    @patch('src.models.time_series.ARIMA')
    def test_train_success(self, mock_arima_class, arima_model, sample_data):
        """Test successful model training."""
        # Mock ARIMA model
        mock_model = Mock()
        mock_model.aic = 1000.0
        mock_model.bic = 1010.0
        mock_model.fit.return_value = mock_model
        mock_arima_class.return_value = mock_model
        
        result = arima_model.train(sample_data)
        
        assert result is not None
        assert 'order' in result
        assert 'seasonal_order' in result
        assert 'aic' in result
        assert arima_model.is_trained
    
    def test_train_insufficient_data(self, arima_model):
        """Test training with insufficient data."""
        short_data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ModelError):
            arima_model.train(short_data)
    
    def test_auto_select_params(self, arima_model, sample_data):
        """Test automatic parameter selection."""
        order, seasonal_order = arima_model._auto_select_params(sample_data)
        
        assert len(order) == 3
        assert len(seasonal_order) == 4
        assert all(isinstance(x, int) for x in order)
        assert all(isinstance(x, int) for x in seasonal_order)
    
    def test_forecast_untrained(self, arima_model):
        """Test forecasting with untrained model."""
        with pytest.raises(ModelError):
            arima_model.forecast(24)
    
    @patch('src.models.time_series.ARIMA')
    def test_forecast_success(self, mock_arima_class, arima_model, sample_data):
        """Test successful forecasting."""
        # Mock trained model
        mock_model = Mock()
        mock_model.aic = 1000.0
        mock_model.fit.return_value = mock_model
        
        # Mock forecast result
        mock_forecast_result = Mock()
        mock_forecast_result.predicted_mean = pd.Series([25, 30, 35, 40, 45])
        mock_model.get_forecast.return_value = mock_forecast_result
        
        mock_arima_class.return_value = mock_model
        arima_model.train(sample_data)
        
        forecast = arima_model.forecast(5)
        
        assert len(forecast) == 5
        assert isinstance(forecast, np.ndarray)
    
    def test_forecast_with_confidence(self, arima_model, sample_data):
        """Test forecasting with confidence intervals."""
        # Mock trained model
        arima_model.model = Mock()
        arima_model.is_trained = True
        
        mock_forecast_result = Mock()
        mock_forecast_result.predicted_mean = pd.Series([25, 30, 35])
        mock_conf_int = pd.DataFrame({
            'lower': [20, 25, 30],
            'upper': [30, 35, 40]
        })
        mock_forecast_result.conf_int.return_value = mock_conf_int
        arima_model.model.get_forecast.return_value = mock_forecast_result
        
        result = arima_model.forecast_with_confidence(3)
        
        assert 'forecast' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert len(result['forecast']) == 3
    
    def test_evaluate(self, arima_model, sample_data):
        """Test model evaluation."""
        # Mock trained model
        arima_model.is_trained = True
        arima_model.forecast = Mock(return_value=np.array([25, 30, 35, 40, 45]))
        
        test_data = pd.Series([20, 25, 30, 35, 40])
        metrics = arima_model.evaluate(test_data)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
    
    def test_save_load(self, arima_model, sample_data, tmp_path):
        """Test model saving and loading."""
        # Mock trained model
        arima_model.model = Mock()
        arima_model.is_trained = True
        arima_model.metadata = {"test": "metadata"}
        
        save_path = tmp_path / "test_model.pkl"
        
        with patch('joblib.dump') as mock_dump:
            arima_model.save(str(save_path))
            mock_dump.assert_called_once()
        
        with patch('joblib.load') as mock_load:
            mock_load.return_value = {
                "model": Mock(),
                "metadata": {"test": "metadata"},
                "name": "ARIMA"
            }
            arima_model.load(str(save_path))
            assert arima_model.is_trained


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        values = np.random.normal(50, 15, 200) + np.sin(np.arange(200) * 0.1) * 20
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance."""
        return LSTMModel(sequence_length=24)
    
    def test_initialization(self, lstm_model):
        """Test model initialization."""
        assert lstm_model.name == "LSTM"
        assert lstm_model.sequence_length == 24
        assert not lstm_model.is_trained
    
    @patch('src.models.time_series.TF_AVAILABLE', True)
    @patch('src.models.time_series.Sequential')
    @patch('src.models.time_series.Adam')
    def test_train_success(self, mock_adam, mock_sequential, lstm_model, sample_data):
        """Test successful model training."""
        # Mock Keras components
        mock_model = Mock()
        mock_model.fit.return_value.history = {'loss': [0.1, 0.05], 'val_loss': [0.15, 0.08]}
        mock_sequential.return_value = mock_model
        
        result = lstm_model.train(sample_data, epochs=2)
        
        assert result is not None
        assert 'sequence_length' in result
        assert 'epochs' in result
        assert 'training_samples' in result
        assert lstm_model.is_trained
    
    def test_train_insufficient_data(self, lstm_model):
        """Test training with insufficient data."""
        short_data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ModelError):
            lstm_model.train(short_data)
    
    def test_create_sequences(self, lstm_model):
        """Test sequence creation for LSTM."""
        data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        X, y = lstm_model._create_sequences(data)
        
        assert X.shape[0] == data.shape[0] - lstm_model.sequence_length
        assert X.shape[1] == lstm_model.sequence_length
        assert X.shape[2] == 1
        assert len(y) == X.shape[0]
    
    @patch('src.models.time_series.TF_AVAILABLE', False)
    def test_tensorflow_unavailable(self):
        """Test LSTM initialization when TensorFlow is unavailable."""
        with pytest.raises(ModelError):
            LSTMModel()


class TestTimeSeriesForecaster:
    """Test cases for TimeSeriesForecaster."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster instance."""
        return TimeSeriesForecaster()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        values = np.random.normal(50, 15, 200) + np.sin(np.arange(200) * 0.1) * 20
        return pd.Series(values, index=dates)
    
    def test_initialization(self, forecaster):
        """Test forecaster initialization."""
        assert forecaster.models == {}
        assert forecaster.model_registry == {}
    
    def test_add_model(self, forecaster):
        """Test adding models to forecaster."""
        mock_model = Mock()
        mock_model.name = "TestModel"
        
        forecaster.add_model("test_model", mock_model)
        
        assert "test_model" in forecaster.models
        assert forecaster.models["test_model"] == mock_model
    
    @patch('src.models.time_series.ARIMAModel')
    def test_train_arima_model(self, mock_arima_class, forecaster, sample_data):
        """Test training ARIMA model through forecaster."""
        mock_model = Mock()
        mock_model.name = "ARIMA"
        mock_model.train.return_value = {"aic": 1000}
        mock_arima_class.return_value = mock_model
        
        result = forecaster.train_model("arima", sample_data)
        
        assert result["model_type"] == "arima"
        assert "model_name" in result
        assert "metadata" in result
        assert len(forecaster.models) == 1
    
    @patch('src.models.time_series.LSTMModel')
    def test_train_lstm_model(self, mock_lstm_class, forecaster, sample_data):
        """Test training LSTM model through forecaster."""
        mock_model = Mock()
        mock_model.name = "LSTM"
        mock_model.train.return_value = {"epochs": 50}
        mock_lstm_class.return_value = mock_model
        
        with patch('src.models.time_series.TF_AVAILABLE', True):
            result = forecaster.train_model("lstm", sample_data)
        
        assert result["model_type"] == "lstm"
        assert "model_name" in result
        assert "metadata" in result
    
    def test_train_unsupported_model(self, forecaster, sample_data):
        """Test training unsupported model type."""
        with pytest.raises(ModelError):
            forecaster.train_model("unsupported", sample_data)
    
    def test_forecast(self, forecaster, sample_data):
        """Test forecasting through forecaster."""
        mock_model = Mock()
        mock_model.forecast.return_value = np.array([25, 30, 35])
        
        forecaster.add_model("test_model", mock_model)
        
        forecast = forecaster.forecast("test_model", 3)
        
        assert len(forecast) == 3
        mock_model.forecast.assert_called_once_with(3)
    
    def test_forecast_nonexistent_model(self, forecaster):
        """Test forecasting with non-existent model."""
        with pytest.raises(ModelError):
            forecaster.forecast("nonexistent", 24)
    
    def test_evaluate_model(self, forecaster, sample_data):
        """Test model evaluation through forecaster."""
        mock_model = Mock()
        mock_model.evaluate.return_value = {"mae": 5.0, "r2": 0.8}
        
        forecaster.add_model("test_model", mock_model)
        
        metrics = forecaster.evaluate_model("test_model", sample_data)
        
        assert metrics["mae"] == 5.0
        assert metrics["r2"] == 0.8
    
    def test_list_models(self, forecaster):
        """Test listing models."""
        mock_model1 = Mock()
        mock_model1.name = "ARIMA"
        mock_model1.is_trained = True
        mock_model1.metadata = {"aic": 1000}
        
        mock_model2 = Mock()
        mock_model2.name = "LSTM"
        mock_model2.is_trained = False
        mock_model2.metadata = {}
        
        forecaster.add_model("model1", mock_model1)
        forecaster.add_model("model2", mock_model2)
        
        models = forecaster.list_models()
        
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models
        assert models["model1"]["type"] == "ARIMA"
        assert models["model1"]["trained"] == True


if __name__ == "__main__":
    pytest.main([__file__])
