"""Data schemas for AirSense system."""

# Standard library imports
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Spark Schemas
# ---------------------------------------------------------------------------

class AirQualitySchema:
    """Schema definitions for air quality data."""

    @staticmethod
    def raw_data_schema() -> StructType:
        """Schema for raw input data."""
        return StructType([
            StructField("datetime", StringType(), True),
            StructField("PM2.5", DoubleType(), True),
            StructField("PM10", DoubleType(), True),
            StructField("NO2", DoubleType(), True),
            StructField("SO2", DoubleType(), True),
            StructField("CO", DoubleType(), True),
            StructField("O3", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("pressure", DoubleType(), True),
            StructField("wind_speed", DoubleType(), True),
            StructField("wind_direction", DoubleType(), True),
        ])

    @staticmethod
    def processed_data_schema() -> StructType:
        """Schema for processed data with engineered features."""
        base_fields = AirQualitySchema.raw_data_schema().fields

        # Add engineered features
        engineered_fields = [
            StructField("hour", DoubleType(), True),
            StructField("dayofweek", DoubleType(), True),
            StructField("month", DoubleType(), True),
            StructField("year", DoubleType(), True),
            StructField("AQI", DoubleType(), True),
            StructField("PM_ratio", DoubleType(), True),
            StructField("NO2_SO2_ratio", DoubleType(), True),
            StructField("temp_humidity_interaction", DoubleType(), True),
            StructField("wind_temp_interaction", DoubleType(), True),
            StructField("scaled_features", StringType(), True),  # Vector stored as string
        ]

        return StructType(base_fields + engineered_fields)


# ---------------------------------------------------------------------------
# Pydantic v2 Models for API
# ---------------------------------------------------------------------------

class AirQualityReading(BaseModel):
    """Single air quality reading."""

    model_config = ConfigDict(populate_by_name=True)

    datetime: datetime
    pm25: Optional[float] = Field(None, alias="PM2.5")
    pm10: Optional[float] = Field(None, alias="PM10")
    no2: Optional[float] = Field(None, alias="NO2")
    so2: Optional[float] = Field(None, alias="SO2")
    co: Optional[float] = Field(None, alias="CO")
    o3: Optional[float] = Field(None, alias="O3")
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None

    @field_validator("pm25", "pm10", "no2", "so2", "co", "o3")
    @classmethod
    def validate_pollutants(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Pollutant values cannot be negative")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < -50 or v > 60):
            raise ValueError("Temperature must be between -50°C and 60°C")
        return v

    @field_validator("humidity")
    @classmethod
    def validate_humidity(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Humidity must be between 0% and 100%")
        return v


class ForecastRequest(BaseModel):
    """Request model for forecasting."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_pollutant": "PM2.5",
                "model_type": "arima",
                "steps": 24,
                "retrain": False,
            }
        }
    )

    target_pollutant: str = Field(..., pattern=r"^(PM2\.5|PM10|NO2|SO2|CO|O3)$")
    model_type: str = Field("arima", pattern=r"^(arima|lstm)$")
    steps: int = Field(24, ge=1, le=168)  # 1 hour to 1 week
    retrain: bool = False


class ForecastDataPoint(BaseModel):
    """A single data point in a forecast response."""

    datetime: str
    value: float
    target_pollutant: str
    model_type: str


class ForecastResponse(BaseModel):
    """Response model for forecasting."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "forecast": [
                    {
                        "datetime": "2024-01-01T13:00:00",
                        "value": 25.5,
                        "target_pollutant": "PM2.5",
                        "model_type": "arima",
                    }
                ],
                "model_name": "arima_PM2.5_20240101_120000",
                "target_pollutant": "PM2.5",
                "steps": 24,
                "generated_at": "2024-01-01T12:00:00",
            }
        }
    )

    forecast: List[Dict[str, Any]]
    model_name: str
    target_pollutant: str
    steps: int
    generated_at: datetime


class AQILevel(str, Enum):
    """AQI level categories."""

    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"


class AQIResponse(BaseModel):
    """Response model for AQI information."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "aqi": 75.0,
                "level": "Moderate",
                "health_impact": "Acceptable for most people",
                "recommendations": "Unusually sensitive people should reduce prolonged outdoor exertion",
                "timestamp": "2024-01-01T12:00:00",
                "component_aqi": {
                    "PM2.5": 65.0,
                    "PM10": 70.0,
                    "NO2": 45.0,
                },
            }
        }
    )

    aqi: float
    level: str
    health_impact: str
    recommendations: str
    timestamp: datetime
    component_aqi: Dict[str, float]


class DataQueryRequest(BaseModel):
    """Request model for data queries."""

    limit: Optional[int] = Field(1000, ge=1, le=10000)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    pollutants: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "DataQueryRequest":
        if self.end_date and self.start_date and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class PipelineStatus(BaseModel):
    """Pipeline status response."""

    status: str
    timestamp: datetime
    processed_files: List[str]
    error_count: int
    warning_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    models_loaded: int
    data_available: bool


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class ModelRegistryEntry(BaseModel):
    """Schema for a model registry entry returned by the API."""

    model_id: str
    model_name: str
    model_type: str
    version: str
    file_path: str
    training_date: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    target_pollutant: Optional[str] = None
    training_data_source: Optional[str] = None
    feature_set: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime


class EvaluationRequest(BaseModel):
    """Request model for model evaluation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "arima_PM2.5_20240101_120000",
                "test_data_path": "data/processed/test.parquet",
                "target_pollutant": "PM2.5",
                "forecast_steps": 24,
            }
        }
    )

    model_id: str
    test_data_path: str
    target_pollutant: str = Field(..., pattern=r"^(PM2\.5|PM10|NO2|SO2|CO|O3)$")
    forecast_steps: int = Field(24, ge=1, le=168)


class EvaluationResponse(BaseModel):
    """Response model for model evaluation."""

    model_id: str
    model_name: str
    model_type: str
    metrics: Dict[str, float]
    test_size: int
    evaluation_date: datetime
    additional_info: Optional[Dict[str, Any]] = None
