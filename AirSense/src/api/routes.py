"""API routes for AirSense."""

# Standard library imports
import os
from datetime import datetime, timedelta
from typing import List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

# Local imports
from ..core.config import get_settings
from ..core.exceptions import ModelError, ValidationError
from ..core.logging import get_logger
from ..data.correlation_analysis import CorrelationAnalyzer
from ..data.schemas import (
    AQIResponse,
    DataQueryRequest,
    ForecastRequest,
    ForecastResponse,
    PipelineStatus,
)
from ..data.time_analysis import TimeBasedAnalyzer
from ..models.simple_forecasting import SimpleForecaster

# Get settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Create router
router = APIRouter()

# Dependency to get app state
async def get_forecaster(request: Request):
    """Get forecaster from app state."""
    app = request.app
    
    if not hasattr(app.state, 'forecaster'):
        raise HTTPException(status_code=503, detail="Forecaster not initialized")
    
    return app.state.forecaster

async def get_data_processor(request: Request):
    """Get data processor from app state."""
    app = request.app
    
    if not hasattr(app.state, 'data_processor'):
        raise HTTPException(status_code=503, detail="Data processor not initialized")
    
    return app.state.data_processor


@router.get("/data", response_model=dict)
async def get_data(
    limit: int = Query(1000, ge=1, le=10000),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    pollutants: Optional[List[str]] = Query(None),
    processor=Depends(get_data_processor)
):
    """Get air quality data with filtering."""
    try:
        # Get latest processed data
        processed_dir = settings.processed_data_dir
        import os
        
        if not os.path.exists(processed_dir):
            raise HTTPException(status_code=404, detail="No processed data available")
        
        # Find latest parquet file
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Apply filters
        if start_date:
            df = df[df['datetime'] >= start_date]
        
        if end_date:
            df = df[df['datetime'] <= end_date]
        
        if pollutants:
            available_pollutants = [p for p in pollutants if p in df.columns]
            if available_pollutants:
                columns_to_keep = ['datetime'] + available_pollutants
                df = df[columns_to_keep]
        
        # Limit results
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        # Convert to dict
        return {
            "data": df.to_dict('records'),
            "count": len(df),
            "columns": list(df.columns),
            "source_file": latest_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /data endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    forecaster=Depends(get_forecaster)
):
    """Generate air quality forecast."""
    try:
        # Get data for the target pollutant
        processed_dir = settings.processed_data_dir
        import os
        
        if not os.path.exists(processed_dir):
            raise HTTPException(status_code=404, detail="No processed data available")
        
        # Load latest data
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        if request.target_pollutant not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Pollutant {request.target_pollutant} not found"
            )
        
        # Prepare data
        data = df.set_index('datetime')[request.target_pollutant]
        data = data.ffill().bfill()
        
        # Train or use existing model
        model_name = f"{request.model_type}_{request.target_pollutant}"
        
        if request.retrain or model_name not in forecaster.models:
            logger.info(f"Training {request.model_type} model for {request.target_pollutant}")
            
            # Train model with appropriate parameters
            train_params = {}
            if request.model_type == "arima":
                train_params = {"auto_params": True}
            elif request.model_type == "lstm":
                train_params = {
                    "epochs": 50,
                    "batch_size": 32,
                    "validation_split": 0.2
                }
            
            result = forecaster.train_model(
                request.model_type, 
                data, 
                **train_params
            )
            model_name = result["model_name"]
        
        # Generate forecast
        forecast_values = forecaster.forecast(model_name, request.steps)
        
        # Create forecast dates
        last_date = df['datetime'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=request.steps,
            freq='H'
        )
        
        # Prepare forecast data
        forecast_data = []
        for date, value in zip(forecast_dates, forecast_values):
            forecast_data.append({
                "datetime": date.isoformat(),
                "value": float(value),
                "target_pollutant": request.target_pollutant,
                "model_type": request.model_type
            })
        
        return ForecastResponse(
            forecast=forecast_data,
            model_name=model_name,
            target_pollutant=request.target_pollutant,
            steps=request.steps,
            generated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aqi", response_model=AQIResponse)
async def get_aqi(processor=Depends(get_data_processor)):
    """Get current AQI and health recommendations."""
    try:
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        
        if not os.path.exists(processed_dir):
            raise HTTPException(status_code=404, detail="No processed data available")
        
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Get latest row
        latest_row = df.iloc[-1]
        
        # Calculate AQI components
        pm25_aqi = _calculate_aqi_pm25(latest_row.get('PM2.5', 0))
        pm10_aqi = _calculate_aqi_pm10(latest_row.get('PM10', 0))
        no2_aqi = _calculate_aqi_no2(latest_row.get('NO2', 0))
        
        overall_aqi = max(pm25_aqi, pm10_aqi, no2_aqi)
        
        return AQIResponse(
            aqi=overall_aqi,
            level=_get_aqi_level(overall_aqi),
            health_impact=_get_health_impact(overall_aqi),
            recommendations=_get_recommendations(overall_aqi),
            timestamp=latest_row['datetime'],
            component_aqi={
                "PM2.5": pm25_aqi,
                "PM10": pm10_aqi,
                "NO2": no2_aqi
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /aqi endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics(processor=Depends(get_data_processor)):
    """Get data statistics."""
    try:
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        
        if not os.path.exists(processed_dir):
            raise HTTPException(status_code=404, detail="No processed data available")
        
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Calculate statistics for pollutants
        stats = {}
        for pollutant in settings.pollutant_columns:
            if pollutant in df.columns:
                col_data = df[pollutant].dropna()
                if len(col_data) > 0:
                    stats[pollutant] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median()),
                        "latest": float(col_data.iloc[-1]) if len(col_data) > 0 else None
                    }
        
        return {
            "statistics": stats,
            "data_range": {
                "start": df['datetime'].min().isoformat(),
                "end": df['datetime'].max().isoformat(),
                "total_records": len(df)
            },
            "source_file": latest_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(forecaster=Depends(get_forecaster)):
    """List available trained models."""
    try:
        models = forecaster.list_models()
        return {
            "models": models,
            "total_count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error in /models endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/run", response_model=PipelineStatus)
async def run_pipeline(
    background_tasks: BackgroundTasks,
    processor=Depends(get_data_processor)
):
    """Run the data processing pipeline."""
    try:
        # Add pipeline execution to background tasks
        background_tasks.add_task(_run_pipeline_task, processor)
        
        return PipelineStatus(
            status="running",
            timestamp=datetime.now(),
            processed_files=[],
            error_count=0,
            warning_count=0
        )
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_pipeline_task(processor):
    """Background task to run pipeline."""
    try:
        logger.info("Starting pipeline execution in background")
        
        # Find input file
        input_file = os.path.join(settings.raw_data_dir, "beijing_demo.csv")
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(settings.processed_data_dir, f"processed_{timestamp}.parquet")
        
        # Run pipeline
        result = processor.process_pipeline(input_file, output_path)
        
        logger.info("Pipeline completed", **result)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


# AQI calculation helper functions
def _calculate_aqi_pm25(pm25):
    """Calculate AQI for PM2.5."""
    if pm25 <= 12: return (50 / 12) * pm25
    elif pm25 <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    else: return 200

def _calculate_aqi_pm10(pm10):
    """Calculate AQI for PM10."""
    if pm10 <= 54: return (50 / 54) * pm10
    elif pm10 <= 154: return ((100 - 51) / (154 - 55)) * (pm10 - 55) + 51
    elif pm10 <= 254: return ((150 - 101) / (254 - 155)) * (pm10 - 155) + 101
    else: return 200

def _calculate_aqi_no2(no2):
    """Calculate AQI for NO2."""
    if no2 <= 53: return (50 / 53) * no2
    elif no2 <= 100: return ((100 - 51) / (100 - 54)) * (no2 - 54) + 51
    elif no2 <= 360: return ((150 - 101) / (360 - 101)) * (no2 - 101) + 101
    else: return 200

def _get_aqi_level(aqi):
    """Get AQI level description."""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    else: return "Unhealthy"

def _get_health_impact(aqi):
    """Get health impact description."""
    if aqi <= 50: return "Air quality is satisfactory"
    elif aqi <= 100: return "Acceptable for most people"
    elif aqi <= 150: return "Sensitive groups may experience health effects"
    else: return "Everyone may experience health effects"

def _get_recommendations(aqi):
    """Get health recommendations."""
    if aqi <= 50: return "Enjoy outdoor activities"
    elif aqi <= 100: return "Unusually sensitive people should reduce prolonged outdoor exertion"
    elif aqi <= 150: return "Sensitive groups should reduce prolonged outdoor exertion"
    else: return "Everyone should avoid prolonged outdoor exertion"


# Simple forecasting endpoints
@router.post("/forecast/simple")
async def simple_forecast(
    request: dict,
    processor=Depends(get_data_processor)
):
    """Generate simple forecast using moving averages and other basic methods."""
    try:
        pollutant = request.get("pollutant", "PM2.5")
        method = request.get("method", "moving_average")
        forecast_steps = request.get("forecast_steps", 24)
        window_size = request.get("window_size", 24)
        
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        if pollutant not in df.columns:
            raise HTTPException(status_code=400, detail=f"Pollutant {pollutant} not found")
        
        # Get time series data
        ts_data = df[pollutant].dropna()
        
        # Initialize simple forecaster
        forecaster = SimpleForecaster()
        
        # Generate forecast based on method
        if method == "moving_average":
            result = forecaster.moving_average_forecast(ts_data, window_size, forecast_steps)
        elif method == "weighted_moving_average":
            result = forecaster.weighted_moving_average_forecast(ts_data, window_size, forecast_steps)
        elif method == "seasonal_naive":
            result = forecaster.seasonal_naive_forecast(ts_data, 24, forecast_steps)
        elif method == "exponential_smoothing":
            alpha = request.get("alpha", 0.3)
            result = forecaster.exponential_smoothing_forecast(ts_data, alpha, forecast_steps)
        elif method == "ensemble":
            result = forecaster.ensemble_forecast(ts_data, forecast_steps)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown forecasting method: {method}")
        
        return {
            "pollutant": pollutant,
            "method": method,
            "forecast_steps": forecast_steps,
            "data_points_used": len(ts_data),
            "forecast": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Time-based analysis endpoints
@router.get("/analysis/time-patterns")
async def time_patterns_analysis(
    pollutant: str = Query(...),
    analysis_type: str = Query("comprehensive"),
    processor=Depends(get_data_processor)
):
    """Analyze time-based patterns for pollutants."""
    try:
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Convert to Spark DataFrame for analysis
        spark_df = processor.spark.createDataFrame(df)
        
        # Initialize analyzer
        analyzer = TimeBasedAnalyzer(processor.spark)
        
        # Perform analysis
        if analysis_type == "daily":
            result = analyzer.daily_patterns(spark_df, pollutant)
        elif analysis_type == "weekly":
            result = analyzer.weekly_patterns(spark_df, pollutant)
        elif analysis_type == "monthly":
            result = analyzer.monthly_patterns(spark_df, pollutant)
        elif analysis_type == "yearly":
            result = analyzer.yearly_trends(spark_df, pollutant)
        elif analysis_type == "comprehensive":
            result = analyzer.comprehensive_time_analysis(spark_df, [pollutant])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")
        
        return {
            "pollutant": pollutant,
            "analysis_type": analysis_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time patterns analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Correlation analysis endpoints
@router.get("/analysis/correlations")
async def correlation_analysis(
    analysis_type: str = Query("comprehensive"),
    processor=Depends(get_data_processor)
):
    """Perform correlation analysis."""
    try:
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Convert to Spark DataFrame
        spark_df = processor.spark.createDataFrame(df)
        
        # Initialize analyzer
        analyzer = CorrelationAnalyzer(processor.spark)
        
        # Perform analysis
        if analysis_type == "pollutants":
            result = analyzer.pollutant_correlations(spark_df)
        elif analysis_type == "weather":
            result = analyzer.weather_pollutant_correlations(spark_df)
        elif analysis_type == "comprehensive":
            result = analyzer.comprehensive_correlation_analysis(spark_df)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")
        
        return {
            "analysis_type": analysis_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/temporal-correlations")
async def temporal_correlations(
    pollutant: str = Query(...),
    processor=Depends(get_data_processor)
):
    """Analyze temporal correlations for a specific pollutant."""
    try:
        # Get latest data
        processed_dir = settings.processed_data_dir
        import os
        parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No processed data available")
        
        latest_file = sorted(parquet_files)[-1]
        df = pd.read_parquet(os.path.join(processed_dir, latest_file))
        
        # Convert to Spark DataFrame
        spark_df = processor.spark.createDataFrame(df)
        
        # Initialize analyzer
        analyzer = CorrelationAnalyzer(processor.spark)
        
        # Perform temporal correlation analysis
        result = analyzer.temporal_correlations(spark_df, pollutant)
        
        return {
            "pollutant": pollutant,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
