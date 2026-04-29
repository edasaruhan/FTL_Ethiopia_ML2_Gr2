"""Custom exceptions for AirSense system."""


class AirSenseError(Exception):
    """Base exception for AirSense system."""
    pass


class DataProcessingError(AirSenseError):
    """Exception raised during data processing operations."""
    pass


class ModelError(AirSenseError):
    """Exception raised during ML model operations."""
    pass


class ConfigurationError(AirSenseError):
    """Exception raised for configuration issues."""
    pass


class APIError(AirSenseError):
    """Exception raised during API operations."""
    pass


class ValidationError(AirSenseError):
    """Exception raised during data validation."""
    pass


class SparkError(DataProcessingError):
    """Exception raised during Spark operations."""
    pass


class ForecastError(ModelError):
    """Exception raised during forecasting operations."""
    pass


class DatabaseError(AirSenseError):
    """Exception raised during database operations."""
    pass
