"""Advanced logging configuration for AirSense."""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from datetime import datetime

from .config import get_settings


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        record.levelname = f"{log_color}{record.levelname}{reset}"
        record.name = f"{log_color}{record.name}{reset}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = True
) -> None:
    """Setup comprehensive logging configuration."""
    
    settings = get_settings()
    
    # Configure structlog
    if enable_structured:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(settings.logs_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self):
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                duration_seconds=duration,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                "Function execution failed",
                function=func.__name__,
                duration_seconds=duration,
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise
    
    return wrapper


def log_data_quality(data, name: str = "dataset"):
    """Log data quality metrics."""
    logger = get_logger("data_quality")
    
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            logger.info(
                "Data quality metrics",
                dataset=name,
                rows=len(data),
                columns=len(data.columns),
                null_values=data.isnull().sum().sum(),
                memory_usage_mb=data.memory_usage(deep=True).sum() / 1024 / 1024,
                duplicate_rows=data.duplicated().sum()
            )
            
            # Log column-specific info
            for col in data.columns:
                null_count = data[col].isnull().sum()
                unique_count = data[col].nunique()
                
                logger.debug(
                    "Column statistics",
                    dataset=name,
                    column=col,
                    null_count=null_count,
                    unique_count=unique_count,
                    dtype=str(data[col].dtype)
                )
        
    except Exception as e:
        logger.warning(f"Failed to log data quality for {name}: {e}")


# Initialize logging on import
settings = get_settings()
setup_logging(
    level=settings.log_level,
    log_file=f"airsense_{datetime.now().strftime('%Y%m%d')}.log",
    enable_console=True,
    enable_structured=True
)
