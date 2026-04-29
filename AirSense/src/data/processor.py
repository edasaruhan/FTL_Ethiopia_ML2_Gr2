"""Advanced Spark data processor for AirSense."""

# Standard library imports
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    cos,
    dayofweek,
    hour,
    lag,
    mean as _mean,
    month,
    sin,
    stddev as _stddev,
    to_timestamp,
    unix_timestamp,
    when,
    year,
)
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

# Local imports
from ..core.config import get_settings
from ..core.exceptions import DataProcessingError, SparkError
from ..core.logging import get_logger, log_data_quality, log_performance


class SparkDataProcessor:
    """Enterprise-grade Spark data processor with advanced capabilities."""
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize or use provided Spark session
        self.spark = spark_session or self._create_spark_session()
        
        # Cache for performance
        self._schema_cache = {}
        
    def _create_spark_session(self) -> SparkSession:
        """Create optimized Spark session."""
        try:
            self.logger.info("Creating Spark session", config=self.settings.spark_config)
            
            builder = SparkSession.builder.appName(self.settings.spark_app_name)
            
            # Apply Spark configuration
            for key, value in self.settings.spark_config.items():
                builder = builder.config(key, value)
            
            spark = builder.getOrCreate()
            
            # Set log level
            spark.sparkContext.setLogLevel("WARN")
            
            self.logger.info("Spark session created successfully")
            return spark
            
        except Exception as e:
            self.logger.error("Failed to create Spark session", error=str(e))
            raise SparkError(f"Failed to create Spark session: {e}")
    
    @property
    def schema(self) -> StructType:
        """Get data schema with caching."""
        if 'air_quality_schema' not in self._schema_cache:
            self._schema_cache['air_quality_schema'] = StructType([
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
                StructField("wind_direction", DoubleType(), True)
            ])
        
        return self._schema_cache['air_quality_schema']
    
    @log_performance
    def load_data(self, file_path: str, schema: Optional[StructType] = None) -> DataFrame:
        """Load data with advanced validation and error handling."""
        try:
            self.logger.info("Loading data", file_path=file_path)
            
            # Use default schema if none provided
            if schema is None:
                schema = self.schema
            
            # Load data
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "false") \
                .option("nullValue", "") \
                .option("treatEmptyValuesAsNulls", "true") \
                .schema(schema) \
                .csv(file_path)
            
            # Log initial data quality
            log_data_quality(df.toPandas(), "raw_data")
            
            self.logger.info("Data loaded successfully", 
                           count=df.count(), 
                           columns=len(df.columns))
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to load data", file_path=file_path, error=str(e))
            raise DataProcessingError(f"Failed to load data from {file_path}: {e}")
    
    @log_performance
    def clean_data(self, df: DataFrame) -> DataFrame:
        """Advanced data cleaning with statistical methods."""
        try:
            self.logger.info("Starting data cleaning")
            
            # Convert datetime
            df = df.withColumn("datetime", to_timestamp(col("datetime"), "yyyy-MM-dd HH:mm:ss"))
            
            # Remove invalid datetime records
            df = df.filter(col("datetime").isNotNull())
            
            # Handle missing values for pollutants
            for pollutant in self.settings.pollutant_columns:
                if pollutant in df.columns:
                    # Forward fill then backward fill
                    window = Window.orderBy("datetime")
                    df = df.withColumn(
                        pollutant,
                        when(col(pollutant).isNull(), 
                              lag(pollutant, 1).over(window))
                        .otherwise(col(pollutant))
                    )
                    
                    # Backward fill for remaining nulls
                    df = df.withColumn(
                        pollutant,
                        when(col(pollutant).isNull(), 
                              _mean(pollutant).over(Window.rowsBetween(-sys.maxsize, sys.maxsize)))
                        .otherwise(col(pollutant))
                    )
            
            # Remove outliers using IQR method
            for pollutant in self.settings.pollutant_columns:
                if pollutant in df.columns:
                    # Calculate quartiles
                    quantiles = df.approxQuantile(pollutant, [0.25, 0.75], 0.05)
                    if len(quantiles) == 2:
                        q1, q3 = quantiles
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        df = df.filter(
                            (col(pollutant) >= lower_bound) & 
                            (col(pollutant) <= upper_bound)
                        )
            
            # Remove duplicates
            df = df.dropDuplicates(["datetime"])
            
            # Log cleaning results
            log_data_quality(df.toPandas(), "cleaned_data")
            
            self.logger.info("Data cleaning completed", count=df.count())
            return df
            
        except Exception as e:
            self.logger.error("Data cleaning failed", error=str(e))
            raise DataProcessingError(f"Data cleaning failed: {e}")
    
    @log_performance
    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """Advanced feature engineering for ML models."""
        try:
            self.logger.info("Starting feature engineering")
            
            # Time-based features
            df = df.withColumn("hour", hour(col("datetime")))
            df = df.withColumn("dayofweek", dayofweek(col("datetime")))
            df = df.withColumn("month", month(col("datetime")))
            df = df.withColumn("year", year(col("datetime")))
            
            # Cyclical encoding for time features
            df = df.withColumn("hour_sin", sin(col("hour") * 2 * 3.14159 / 24))
            df = df.withColumn("hour_cos", cos(col("hour") * 2 * 3.14159 / 24))
            df = df.withColumn("dayofweek_sin", sin(col("dayofweek") * 2 * 3.14159 / 7))
            df = df.withColumn("dayofweek_cos", cos(col("dayofweek") * 2 * 3.14159 / 7))
            
            # Air Quality Index calculation
            df = df.withColumn("AQI", 
                (col("PM2.5") * 0.4 + col("PM10") * 0.3 + 
                 col("NO2") * 0.2 + col("SO2") * 0.1))
            
            # Pollutant ratios
            df = df.withColumn("PM_ratio", 
                col("PM2.5") / (col("PM10") + 0.001))
            df = df.withColumn("NO2_SO2_ratio", 
                col("NO2") / (col("SO2") + 0.001))
            
            # Weather interaction features
            if all(col in df.columns for col in ["temperature", "humidity"]):
                df = df.withColumn("temp_humidity_interaction", 
                    col("temperature") * col("humidity"))
            
            if all(col in df.columns for col in ["wind_speed", "temperature"]):
                df = df.withColumn("wind_temp_interaction", 
                    col("wind_speed") * col("temperature"))
            
            # Lag features for time series
            window = Window.orderBy("datetime")
            for pollutant in self.settings.pollutant_columns[:3]:  # Top 3 pollutants
                if pollutant in df.columns:
                    df = df.withColumn(f"{pollutant}_lag_1h", 
                        lag(col(pollutant), 1).over(window))
                    df = df.withColumn(f"{pollutant}_lag_24h", 
                        lag(col(pollutant), 24).over(window))
            
            # Rolling averages
            for pollutant in self.settings.pollutant_columns[:3]:
                if pollutant in df.columns:
                    df = df.withColumn(f"{pollutant}_rolling_24h",
                        _mean(col(pollutant)).over(Window.rowsBetween(-23, 0)))
            
            self.logger.info("Feature engineering completed", 
                           features=len(df.columns))
            return df
            
        except Exception as e:
            self.logger.error("Feature engineering failed", error=str(e))
            raise DataProcessingError(f"Feature engineering failed: {e}")
    
    def create_ml_pipeline(self, feature_columns: List[str]) -> Pipeline:
        """Create ML pipeline with feature scaling."""
        try:
            assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol="features",
                handleInvalid="skip"
            )
            
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withMean=True,
                withStd=True
            )
            
            pipeline = Pipeline(stages=[assembler, scaler])
            
            self.logger.info("ML pipeline created", 
                           feature_count=len(feature_columns))
            return pipeline
            
        except Exception as e:
            self.logger.error("Failed to create ML pipeline", error=str(e))
            raise DataProcessingError(f"Failed to create ML pipeline: {e}")
    
    @log_performance
    def process_pipeline(self, 
                        input_path: str, 
                        output_path: str,
                        feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Complete data processing pipeline."""
        try:
            self.logger.info("Starting complete processing pipeline")
            
            # Load and clean data
            df = self.load_data(input_path)
            df = self.clean_data(df)
            df = self.feature_engineering(df)
            
            # Define feature columns if not provided
            if feature_columns is None:
                feature_columns = [
                    "hour", "dayofweek", "month", "temperature", "humidity", 
                    "pressure", "wind_speed", "wind_direction", "PM_ratio", 
                    "NO2_SO2_ratio", "temp_humidity_interaction", "wind_temp_interaction",
                    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"
                ]
                
                # Add lag features if they exist
                existing_cols = df.columns
                lag_cols = [col for col in existing_cols if "_lag_" in col]
                feature_columns.extend(lag_cols)
                
                # Filter to only existing columns
                feature_columns = [col for col in feature_columns if col in existing_cols]
            
            # Create and apply ML pipeline
            ml_pipeline = self.create_ml_pipeline(feature_columns)
            processed_df = ml_pipeline.fit(df).transform(df)
            
            # Select final columns
            final_columns = ["datetime", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "scaled_features"]
            final_df = processed_df.select(*final_columns)
            
            # Save processed data
            output_dir = Path(output_path)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            
            final_df.write.mode("overwrite").parquet(str(output_path))
            
            # Generate summary statistics
            summary = self.generate_summary(final_df)
            
            result = {
                "status": "success",
                "output_path": str(output_path),
                "record_count": final_df.count(),
                "feature_count": len(feature_columns),
                "summary": summary
            }
            
            self.logger.info("Processing pipeline completed successfully", **result)
            return result
            
        except Exception as e:
            self.logger.error("Processing pipeline failed", error=str(e))
            raise DataProcessingError(f"Processing pipeline failed: {e}")
    
    def generate_summary(self, df: DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        try:
            # Convert to pandas for detailed analysis
            pandas_df = df.toPandas()
            
            summary = {
                "basic_stats": {
                    "record_count": len(pandas_df),
                    "column_count": len(pandas_df.columns),
                    "memory_usage_mb": pandas_df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "date_range": {
                    "start": pandas_df['datetime'].min().isoformat() if 'datetime' in pandas_df.columns else None,
                    "end": pandas_df['datetime'].max().isoformat() if 'datetime' in pandas_df.columns else None
                },
                "pollutant_stats": {}
            }
            
            # Pollutant-specific statistics
            for pollutant in self.settings.pollutant_columns:
                if pollutant in pandas_df.columns:
                    col_data = pandas_df[pollutant].dropna()
                    if len(col_data) > 0:
                        summary["pollutant_stats"][pollutant] = {
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "median": float(col_data.median()),
                            "null_count": int(pandas_df[pollutant].isnull().sum())
                        }
            
            return summary
            
        except Exception as e:
            self.logger.warning("Failed to generate summary", error=str(e))
            return {"error": str(e)}
    
    def stop(self) -> None:
        """Stop Spark session."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")

