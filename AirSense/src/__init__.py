"""
AirSense - Big Data Air Quality Analysis & Forecasting System

A production-ready system for air quality monitoring and forecasting
using Apache Spark, advanced ML models, and real-time APIs.
"""

__version__ = "1.0.0"
__author__ = "AirSense Team"
__email__ = "team@airsense.com"

from .core import config
from .core.logging import get_logger

__all__ = ["config", "get_logger"]
