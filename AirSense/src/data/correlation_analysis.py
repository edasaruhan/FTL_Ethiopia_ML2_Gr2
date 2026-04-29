"""Correlation analysis for air quality and weather data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, corr, count, when, isnan
from pyspark.sql.types import DoubleType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

from ..core.config import get_settings
from ..core.logging import get_logger, log_performance
from ..core.exceptions import DataProcessingError


class CorrelationAnalyzer:
    """Correlation analysis for air quality data and weather variables."""
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.spark = spark_session or SparkSession.builder.appName("CorrelationAnalysis").getOrCreate()
    
    @log_performance
    def pollutant_correlations(self, df: DataFrame) -> Dict[str, Any]:
        """Calculate correlations between different pollutants."""
        try:
            self.logger.info("Calculating pollutant correlations")
            
            # Get available pollutant columns
            available_pollutants = [col for col in self.settings.pollutant_columns if col in df.columns]
            
            if len(available_pollutants) < 2:
                return {"message": "Need at least 2 pollutants for correlation analysis"}
            
            # Create correlation matrix using Spark
            # First, create vector for correlation calculation
            assembler = VectorAssembler(
                inputCols=available_pollutants,
                outputCol="features",
                handleInvalid="skip"
            )
            
            df_vector = assembler.transform(df).select("features")
            
            # Calculate correlation matrix
            correlation_matrix = Correlation.corr(df_vector, "features").collect()[0][0]
            
            # Convert to numpy array for easier processing
            corr_array = correlation_matrix.toArray()
            
            # Create correlation dictionary
            correlations = {}
            for i, pollutant1 in enumerate(available_pollutants):
                correlations[pollutant1] = {}
                for j, pollutant2 in enumerate(available_pollutants):
                    correlations[pollutant1][pollutant2] = float(corr_array[i, j])
            
            # Find strongest correlations (excluding self-correlations)
            strong_correlations = []
            for i, pollutant1 in enumerate(available_pollutants):
                for j, pollutant2 in enumerate(available_pollutants):
                    if i < j:  # Avoid duplicates and self-correlations
                        corr_value = float(corr_array[i, j])
                        if abs(corr_value) > 0.3:  # Moderate correlation threshold
                            strong_correlations.append({
                                "pollutant1": pollutant1,
                                "pollutant2": pollutant2,
                                "correlation": corr_value,
                                "strength": "strong" if abs(corr_value) > 0.7 else "moderate",
                                "direction": "positive" if corr_value > 0 else "negative"
                            })
            
            # Sort by absolute correlation value
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            result = {
                "analysis_type": "pollutant_correlations",
                "pollutants_analyzed": available_pollutants,
                "correlation_matrix": correlations,
                "strong_correlations": strong_correlations,
                "summary": {
                    "total_pairs": len(strong_correlations),
                    "positive_correlations": len([c for c in strong_correlations if c["correlation"] > 0]),
                    "negative_correlations": len([c for c in strong_correlations if c["correlation"] < 0]),
                    "strongest_correlation": strong_correlations[0] if strong_correlations else None
                }
            }
            
            self.logger.info("Pollutant correlation analysis completed",
                           correlations_found=len(strong_correlations))
            return result
            
        except Exception as e:
            self.logger.error("Pollutant correlation analysis failed", error=str(e))
            raise DataProcessingError(f"Pollutant correlation analysis failed: {e}")
    
    @log_performance
    def weather_pollutant_correlations(self, df: DataFrame) -> Dict[str, Any]:
        """Calculate correlations between pollutants and weather variables."""
        try:
            self.logger.info("Calculating weather-pollutant correlations")
            
            # Define weather variables
            weather_vars = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]
            available_weather = [var for var in weather_vars if var in df.columns]
            available_pollutants = [col for col in self.settings.pollutant_columns if col in df.columns]
            
            if not available_weather or not available_pollutants:
                return {"message": "Insufficient weather or pollutant data for correlation analysis"}
            
            # Calculate correlations using Spark
            correlations = {}
            significant_correlations = []
            
            for pollutant in available_pollutants:
                correlations[pollutant] = {}
                for weather_var in available_weather:
                    # Calculate correlation using Spark
                    corr_value = df.select(corr(col(pollutant), col(weather_var))).collect()[0][0]
                    
                    if corr_value is not None and not np.isnan(corr_value):
                        correlations[pollutant][weather_var] = float(corr_value)
                        
                        # Check for significant correlations
                        if abs(corr_value) > 0.2:  # Lower threshold for weather correlations
                            significant_correlations.append({
                                "pollutant": pollutant,
                                "weather_variable": weather_var,
                                "correlation": float(corr_value),
                                "strength": "strong" if abs(corr_value) > 0.5 else "moderate",
                                "direction": "positive" if corr_value > 0 else "negative"
                            })
            
            # Sort by absolute correlation value
            significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Group by weather variable for insights
            weather_insights = {}
            for weather_var in available_weather:
                weather_corrs = [c for c in significant_correlations if c["weather_variable"] == weather_var]
                if weather_corrs:
                    weather_insights[weather_var] = {
                        "most_affected_pollutant": max(weather_corrs, key=lambda x: abs(x["correlation"])),
                        "correlation_count": len(weather_corrs),
                        "positive_correlations": len([c for c in weather_corrs if c["correlation"] > 0]),
                        "negative_correlations": len([c for c in weather_corrs if c["correlation"] < 0])
                    }
            
            result = {
                "analysis_type": "weather_pollutant_correlations",
                "weather_variables": available_weather,
                "pollutants": available_pollutants,
                "correlation_matrix": correlations,
                "significant_correlations": significant_correlations,
                "weather_insights": weather_insights,
                "summary": {
                    "total_correlations": len(significant_correlations),
                    "strong_correlations": len([c for c in significant_correlations if c["strength"] == "strong"]),
                    "most_influential_weather": max(weather_insights.items(), 
                                                  key=lambda x: x[1]["correlation_count"])[0] if weather_insights else None
                }
            }
            
            self.logger.info("Weather-pollutant correlation analysis completed",
                           correlations_found=len(significant_correlations))
            return result
            
        except Exception as e:
            self.logger.error("Weather-pollutant correlation analysis failed", error=str(e))
            raise DataProcessingError(f"Weather-pollutant correlation analysis failed: {e}")
    
    @log_performance
    def temporal_correlations(self, df: DataFrame, pollutant: str) -> Dict[str, Any]:
        """Analyze temporal correlations for a specific pollutant."""
        try:
            self.logger.info("Analyzing temporal correlations", pollutant=pollutant)
            
            if pollutant not in df.columns:
                raise DataProcessingError(f"Pollutant {pollutant} not found in data")
            
            # Convert to pandas for temporal analysis
            pandas_df = df.select("datetime", pollutant).toPandas()
            pandas_df["datetime"] = pd.to_datetime(pandas_df["datetime"])
            pandas_df = pandas_df.sort_values("datetime")
            
            # Remove null values
            pandas_df = pandas_df.dropna()
            
            if len(pandas_df) < 48:  # Need at least 2 days of hourly data
                return {"message": "Insufficient data for temporal correlation analysis"}
            
            # Create lag features
            for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
                pandas_df[f"{pollutant}_lag_{lag}"] = pandas_df[pollutant].shift(lag)
            
            # Calculate autocorrelations
            autocorrelations = {}
            for lag in [1, 24, 168]:
                lag_col = f"{pollutant}_lag_{lag}"
                valid_data = pandas_df[[pollutant, lag_col]].dropna()
                if len(valid_data) > 10:
                    corr_value = valid_data[pollutant].corr(valid_data[lag_col])
                    autocorrelations[f"lag_{lag}h"] = float(corr_value)
            
            # Calculate rolling correlations
            rolling_correlations = {}
            for window in [24, 168]:  # 1 day, 1 week
                rolling_corr = pandas_df[pollutant].rolling(window=window).corr(
                    pandas_df[pollutant].shift(window)
                )
                rolling_correlations[f"rolling_{window}h"] = {
                    "mean": float(rolling_corr.mean()),
                    "std": float(rolling_corr.std()),
                    "max": float(rolling_corr.max()),
                    "min": float(rolling_corr.min())
                }
            
            # Seasonal decomposition insights
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Resample to daily if we have enough data
                if len(pandas_df) >= 720:  # 30 days of hourly data
                    daily_data = pandas_df.set_index("datetime").resample('D')[pollutant].mean()
                    
                    if len(daily_data) >= 30:  # At least 30 days
                        decomposition = seasonal_decompose(daily_data.dropna(), model='additive', period=7)
                        
                        seasonal_strength = np.std(decomposition.seasonal) / np.std(daily_data)
                        trend_strength = np.std(decomposition.trend.dropna()) / np.std(daily_data)
                        
                        seasonal_insights = {
                            "seasonal_strength": float(seasonal_strength),
                            "trend_strength": float(trend_strength),
                            "has_strong_seasonality": seasonal_strength > 0.3,
                            "has_strong_trend": trend_strength > 0.3
                        }
                    else:
                        seasonal_insights = {"message": "Insufficient daily data for seasonal analysis"}
                else:
                    seasonal_insights = {"message": "Insufficient data for seasonal decomposition"}
                    
            except ImportError:
                seasonal_insights = {"message": "statsmodels not available for seasonal analysis"}
            except Exception as e:
                seasonal_insights = {"message": f"Seasonal analysis failed: {str(e)}"}
            
            result = {
                "analysis_type": "temporal_correlations",
                "pollutant": pollutant,
                "data_points": len(pandas_df),
                "autocorrelations": autocorrelations,
                "rolling_correlations": rolling_correlations,
                "seasonal_insights": seasonal_insights,
                "summary": {
                    "strong_autocorrelation": max(abs(v) for v in autocorrelations.values()) if autocorrelations else 0,
                    "persistence_pattern": "high" if autocorrelations.get("lag_24h", 0) > 0.5 else "moderate" if autocorrelations.get("lag_24h", 0) > 0.2 else "low"
                }
            }
            
            self.logger.info("Temporal correlation analysis completed", pollutant=pollutant)
            return result
            
        except Exception as e:
            self.logger.error("Temporal correlation analysis failed", error=str(e))
            raise DataProcessingError(f"Temporal correlation analysis failed: {e}")
    
    @log_performance
    def comprehensive_correlation_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis."""
        try:
            self.logger.info("Starting comprehensive correlation analysis")
            
            results = {}
            
            # Pollutant correlations
            results["pollutant_correlations"] = self.pollutant_correlations(df)
            
            # Weather-pollutant correlations
            results["weather_pollutant_correlations"] = self.weather_pollutant_correlations(df)
            
            # Temporal correlations for main pollutants
            temporal_results = {}
            main_pollutants = ["PM2.5", "PM10", "NO2"]  # Focus on main pollutants
            available_pollutants = [p for p in main_pollutants if p in df.columns]
            
            for pollutant in available_pollutants:
                try:
                    temporal_results[pollutant] = self.temporal_correlations(df, pollutant)
                except Exception as e:
                    self.logger.warning(f"Temporal analysis failed for {pollutant}", error=str(e))
                    temporal_results[pollutant] = {"error": str(e)}
            
            results["temporal_correlations"] = temporal_results
            
            # Generate summary insights
            summary = self._generate_correlation_summary(results)
            
            result = {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "comprehensive_analysis": results,
                "summary_insights": summary
            }
            
            self.logger.info("Comprehensive correlation analysis completed")
            return result
            
        except Exception as e:
            self.logger.error("Comprehensive correlation analysis failed", error=str(e))
            raise DataProcessingError(f"Comprehensive correlation analysis failed: {e}")
    
    def _generate_correlation_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from correlation analyses."""
        try:
            summary = {
                "key_findings": [],
                "weather_impacts": {},
                "pollutant_relationships": {},
                "temporal_patterns": {}
            }
            
            # Pollutant correlation insights
            if "pollutant_correlations" in analyses:
                pollutant_corr = analyses["pollutant_correlations"]
                if "strong_correlations" in pollutant_corr:
                    strong_corr = pollutant_corr["strong_correlations"]
                    if strong_corr:
                        strongest = strong_corr[0]
                        summary["key_findings"].append(
                            f"Strongest correlation: {strongest['pollutant1']} and {strongest['pollutant2']} "
                            f"({strongest['correlation']:.3f}, {strongest['strength']} {strongest['direction']})"
                        )
                        summary["pollutant_relationships"]["strongest_pair"] = {
                            "pollutants": [strongest['pollutant1'], strongest['pollutant2']],
                            "correlation": strongest['correlation'],
                            "relationship": strongest['direction']
                        }
            
            # Weather impact insights
            if "weather_pollutant_correlations" in analyses:
                weather_corr = analyses["weather_pollutant_correlations"]
                if "significant_correlations" in weather_corr:
                    sig_corr = weather_corr["significant_correlations"]
                    if sig_corr:
                        # Most influential weather variable
                        weather_impacts = {}
                        for corr in sig_corr:
                            weather_var = corr["weather_variable"]
                            if weather_var not in weather_impacts:
                                weather_impacts[weather_var] = []
                            weather_impacts[weather_var].append(corr)
                        
                        most_influential = max(weather_impacts.keys(), 
                                             key=lambda x: len(weather_impacts[x]))
                        
                        summary["key_findings"].append(
                            f"Most influential weather factor: {most_influential} "
                            f"(affects {len(weather_impacts[most_influential])} pollutants)"
                        )
                        
                        summary["weather_impacts"] = {
                            "most_influential": most_influential,
                            "affected_pollutants_count": len(weather_impacts[most_influential]),
                            "strongest_impact": max(sig_corr, key=lambda x: abs(x["correlation"]))
                        }
            
            # Temporal pattern insights
            if "temporal_correlations" in analyses:
                temp_corr = analyses["temporal_correlations"]
                high_persistence_pollutants = []
                
                for pollutant, analysis in temp_corr.items():
                    if "summary" in analysis:
                        persistence = analysis["summary"].get("persistence_pattern", "low")
                        if persistence == "high":
                            high_persistence_pollutants.append(pollutant)
                
                if high_persistence_pollutants:
                    summary["key_findings"].append(
                        f"High persistence pollutants: {', '.join(high_persistence_pollutants)}"
                    )
                    summary["temporal_patterns"]["high_persistence"] = high_persistence_pollutants
            
            return summary
            
        except Exception as e:
            self.logger.warning("Failed to generate correlation summary", error=str(e))
            return {"error": str(e)}
