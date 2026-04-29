"""Tests for data processor module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.processor import SparkDataProcessor
from src.core.exceptions import DataProcessingError, SparkError


class TestSparkDataProcessor:
    """Test cases for SparkDataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create test processor instance."""
        with patch('src.data.processor.SparkSession') as mock_spark:
            mock_session = Mock()
            mock_spark.builder.appName.return_value.config.return_value.getOrCreate.return_value = mock_session
            return SparkDataProcessor(mock_session)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        data = {
            'datetime': dates.strftime('%Y-%m-%d %H:%M:%S'),
            'PM2.5': np.random.uniform(10, 100, 100),
            'PM10': np.random.uniform(20, 150, 100),
            'NO2': np.random.uniform(5, 50, 100),
            'SO2': np.random.uniform(2, 30, 100),
            'CO': np.random.uniform(100, 1000, 100),
            'O3': np.random.uniform(10, 80, 100),
            'temperature': np.random.uniform(-10, 30, 100),
            'humidity': np.random.uniform(20, 80, 100),
            'pressure': np.random.uniform(980, 1020, 100),
            'wind_speed': np.random.uniform(0, 10, 100),
            'wind_direction': np.random.uniform(0, 360, 100)
        }
        return pd.DataFrame(data)
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.settings is not None
        assert processor.logger is not None
        assert processor.spark is not None
    
    def test_schema_property(self, processor):
        """Test schema property."""
        schema = processor.schema
        assert schema is not None
        assert 'datetime' in [field.name for field in schema.fields]
        assert 'PM2.5' in [field.name for field in schema.fields]
    
    @patch('pyspark.sql.DataFrame')
    def test_load_data_success(self, mock_df, processor, sample_data):
        """Test successful data loading."""
        # Mock Spark read operations
        processor.spark.read.csv.return_value = mock_df
        mock_df.count.return_value = len(sample_data)
        mock_df.columns = sample_data.columns.tolist()
        
        # Mock toPandas for quality logging
        mock_df.toPandas.return_value = sample_data
        
        result = processor.load_data("test_file.csv")
        
        assert result is not None
        processor.spark.read.csv.assert_called_once()
    
    def test_clean_data(self, processor):
        """Test data cleaning functionality."""
        # Create mock DataFrame with dirty data
        mock_df = Mock()
        mock_df.filter.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.dropDuplicates.return_value = mock_df
        mock_df.count.return_value = 100
        
        result = processor.clean_data(mock_df)
        
        assert result is not None
        # Verify cleaning operations were called
        mock_df.filter.assert_called()
        mock_df.withColumn.assert_called()
        mock_df.dropDuplicates.assert_called()
    
    def test_feature_engineering(self, processor):
        """Test feature engineering."""
        # Create mock DataFrame
        mock_df = Mock()
        mock_df.withColumn.return_value = mock_df
        mock_df.columns = ['datetime', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                         'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
        
        result = processor.feature_engineering(mock_df)
        
        assert result is not None
        # Verify feature engineering operations
        assert mock_df.withColumn.call_count > 0
    
    def test_create_ml_pipeline(self, processor):
        """Test ML pipeline creation."""
        feature_columns = ['hour', 'dayofweek', 'temperature', 'humidity']
        
        with patch('src.data.processor.Pipeline') as mock_pipeline:
            pipeline = processor.create_ml_pipeline(feature_columns)
            
            assert pipeline is not None
            mock_pipeline.assert_called_once()
    
    @patch('src.data.processor.Path')
    def test_process_pipeline_success(self, mock_path, processor, sample_data):
        """Test complete processing pipeline."""
        # Mock all the operations
        processor.load_data = Mock(return_value=Mock())
        processor.clean_data = Mock(return_value=Mock())
        processor.feature_engineering = Mock(return_value=Mock())
        processor.create_ml_pipeline = Mock(return_value=Mock())
        
        mock_ml_pipeline = Mock()
        mock_ml_pipeline.fit.return_value.transform.return_value = Mock()
        mock_ml_pipeline.fit.return_value.transform.return_value.count.return_value = 100
        mock_ml_pipeline.fit.return_value.transform.return_value.columns = ['datetime', 'PM2.5']
        mock_ml_pipeline.fit.return_value.transform.return_value.write.mode.return_value.parquet.return_value = Mock()
        
        processor.create_ml_pipeline.return_value = mock_ml_pipeline
        processor.generate_summary = Mock(return_value={"test": "summary"})
        
        result = processor.process_pipeline("input.csv", "output.parquet")
        
        assert result['status'] == 'success'
        assert 'output_path' in result
        assert 'record_count' in result
    
    def test_generate_summary(self, processor):
        """Test summary generation."""
        # Mock DataFrame
        mock_df = Mock()
        mock_df.toPandas.return_value = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='H'),
            'PM2.5': np.random.uniform(10, 100, 100),
            'PM10': np.random.uniform(20, 150, 100)
        })
        
        summary = processor.generate_summary(mock_df)
        
        assert 'basic_stats' in summary
        assert 'date_range' in summary
        assert 'pollutant_stats' in summary
    
    def test_spark_error_handling(self, processor):
        """Test error handling for Spark operations."""
        # Mock Spark session to raise exception
        processor.spark.read.csv.side_effect = Exception("Spark error")
        
        with pytest.raises(DataProcessingError):
            processor.load_data("test_file.csv")
    
    def test_stop(self, processor):
        """Test stopping Spark session."""
        processor.spark.stop = Mock()
        
        processor.stop()
        
        processor.spark.stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
