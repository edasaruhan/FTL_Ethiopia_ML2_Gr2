"""Configuration management for AirSense."""

import os
from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="AirSense", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Data paths
    data_dir: str = Field(default="data", env="DATA_DIR")
    raw_data_dir: str = Field(default="data/raw", env="RAW_DATA_DIR")
    processed_data_dir: str = Field(default="data/processed", env="PROCESSED_DATA_DIR")
    models_dir: str = Field(default="models", env="MODELS_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    
    # Spark configuration
    spark_app_name: str = Field(default="AirSense", env="SPARK_APP_NAME")
    spark_master: Optional[str] = Field(default=None, env="SPARK_MASTER")
    spark_executor_memory: str = Field(default="4g", env="SPARK_EXECUTOR_MEMORY")
    spark_driver_memory: str = Field(default="2g", env="SPARK_DRIVER_MEMORY")
    spark_executor_cores: int = Field(default=2, env="SPARK_EXECUTOR_CORES")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Database (future)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # ML configuration
    default_forecast_steps: int = Field(default=24, env="DEFAULT_FORECAST_STEPS")
    model_retrain_interval: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL")  # hours
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @property
    def spark_config(self) -> dict:
        """Get Spark configuration as dictionary."""
        config = {
            "spark.app.name": self.spark_app_name,
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        }
        
        if self.spark_master:
            config["spark.master"] = self.spark_master
            
        config.update({
            "spark.executor.memory": self.spark_executor_memory,
            "spark.driver.memory": self.spark_driver_memory,
            "spark.executor.cores": str(self.spark_executor_cores),
        })
        
        return config
    
    @property
    def pollutant_columns(self) -> List[str]:
        """Standard pollutant columns."""
        return ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    
    @property
    def weather_columns(self) -> List[str]:
        """Weather-related columns."""
        return ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.setup_directories()
    return settings
