"""Machine learning models for AirSense."""

from .time_series import TimeSeriesForecaster, ARIMAModel, LSTMModel
from .registry import ModelRegistry

__all__ = [
    "TimeSeriesForecaster",
    "ARIMAModel",
    "LSTMModel",
    "ModelRegistry",
]

# ModelEvaluator is imported lazily to avoid errors when the
# evaluation module has not been implemented yet (task 6).
try:
    from .evaluation import ModelEvaluator  # noqa: F401
    __all__.append("ModelEvaluator")
except ImportError:
    pass
