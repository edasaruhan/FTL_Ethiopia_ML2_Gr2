"""Time-based analysis for air quality data patterns."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, month, year, weekofyear,
    mean as _mean, stddev as _stddev, min as _min, max as _max,
    count, when, isnan
)

from ..core.config import get_settings
from ..core.logging import get_logger, log_performance
from ..core.exceptions import DataProcessingError


class TimeBasedAnalyzer:
    """Time-based pattern analysis for air quality data."""
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.spark = spark_session or SparkSession.builder.appName("TimeAnalysis").getOrCreate()
    
    @log_performance
    def daily_patterns(self, df: DataFrame, pollutant: str) -> Dict[str, Any]:
        """Analyze daily patterns for a specific pollutant."""
        try:
            self.logger.info("Analyzing daily patterns", pollutant=pollutant)
            
            if pollutant not in df.columns:
                raise DataProcessingError(f"Pollutant {pollutant} not found in data")
            
            # Extract hour from datetime
            df_with_hour = df.withColumn("hour", hour(col("datetime")))
            
            # Calculate hourly statistics
            hourly_stats = df_with_hour.groupBy("hour").agg(
                _mean(col(pollutant)).alias("mean"),
                _stddev(col(pollutant)).alias("std"),
                _min(col(pollutant)).alias("min"),
                _max(col(pollutant)).alias("max"),
                count(col(pollutant)).alias("count")
            ).orderBy("hour")
            
            # Convert to pandas for analysis
            hourly_df = hourly_stats.toPandas()
            
            # Find peak hours
            peak_hour = hourly_df.loc[hourly_df['mean'].idxmax()]
            low_hour = hourly_df.loc[hourly_df['mean'].idxmin()]
            
            # Calculate hourly variation coefficient
            hourly_df['cv'] = hourly_df['std'] / hourly_df['mean']
            
            result = {
                "pollutant": pollutant,
                "analysis_type": "daily",
                "peak_hour": {
                    "hour": int(peak_hour['hour']),
                    "mean_value": float(peak_hour['mean']),
                    "max_value": float(peak_hour['max'])
                },
                "low_hour": {
                    "hour": int(low_hour['hour']),
                    "mean_value": float(low_hour['mean']),
                    "min_value": float(low_hour['min'])
                },
                "hourly_stats": hourly_df.to_dict('records'),
                "daily_variation": {
                    "range": float(hourly_df['mean'].max() - hourly_df['mean'].min()),
                    "coefficient_of_variation": float(hourly_df['cv'].mean())
                }
            }
            
            self.logger.info("Daily pattern analysis completed", 
                           peak_hour=int(peak_hour['hour']),
                           low_hour=int(low_hour['hour']))
            return result
            
        except Exception as e:
            self.logger.error("Daily pattern analysis failed", error=str(e))
            raise DataProcessingError(f"Daily pattern analysis failed: {e}")
    
    @log_performance
    def weekly_patterns(self, df: DataFrame, pollutant: str) -> Dict[str, Any]:
        """Analyze weekly patterns for a specific pollutant."""
        try:
            self.logger.info("Analyzing weekly patterns", pollutant=pollutant)
            
            if pollutant not in df.columns:
                raise DataProcessingError(f"Pollutant {pollutant} not found in data")
            
            # Extract day of week (0=Sunday, 6=Saturday)
            df_with_dow = df.withColumn("dayofweek", dayofweek(col("datetime")))
            
            # Calculate daily statistics
            daily_stats = df_with_dow.groupBy("dayofweek").agg(
                _mean(col(pollutant)).alias("mean"),
                _stddev(col(pollutant)).alias("std"),
                _min(col(pollutant)).alias("min"),
                _max(col(pollutant)).alias("max"),
                count(col(pollutant)).alias("count")
            ).orderBy("dayofweek")
            
            # Convert to pandas for analysis
            daily_df = daily_stats.toPandas()
            
            # Map day numbers to names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            daily_df['day_name'] = daily_df['dayofweek'].apply(lambda x: day_names[x % 7])
            
            # Find peak and low days
            peak_day = daily_df.loc[daily_df['mean'].idxmax()]
            low_day = daily_df.loc[daily_df['mean'].idxmin()]
            
            # Weekend vs weekday comparison
            weekend_days = [0, 6]  # Sunday, Saturday
            weekday_days = [1, 2, 3, 4, 5]  # Monday-Friday
            
            weekend_avg = daily_df[daily_df['dayofweek'].isin(weekend_days)]['mean'].mean()
            weekday_avg = daily_df[daily_df['dayofweek'].isin(weekday_days)]['mean'].mean()
            
            result = {
                "pollutant": pollutant,
                "analysis_type": "weekly",
                "peak_day": {
                    "day": peak_day['day_name'],
                    "dayofweek": int(peak_day['dayofweek']),
                    "mean_value": float(peak_day['mean']),
                    "max_value": float(peak_day['max'])
                },
                "low_day": {
                    "day": low_day['day_name'],
                    "dayofweek": int(low_day['dayofweek']),
                    "mean_value": float(low_day['mean']),
                    "min_value": float(low_day['min'])
                },
                "weekend_vs_weekday": {
                    "weekend_average": float(weekend_avg),
                    "weekday_average": float(weekday_avg),
                    "difference": float(weekend_avg - weekday_avg),
                    "percent_difference": float(((weekend_avg - weekday_avg) / weekday_avg) * 100)
                },
                "daily_stats": daily_df[['day_name', 'mean', 'std', 'min', 'max']].to_dict('records')
            }
            
            self.logger.info("Weekly pattern analysis completed",
                           peak_day=peak_day['day_name'],
                           low_day=low_day['day_name'])
            return result
            
        except Exception as e:
            self.logger.error("Weekly pattern analysis failed", error=str(e))
            raise DataProcessingError(f"Weekly pattern analysis failed: {e}")
    
    @log_performance
    def monthly_patterns(self, df: DataFrame, pollutant: str) -> Dict[str, Any]:
        """Analyze monthly patterns for a specific pollutant."""
        try:
            self.logger.info("Analyzing monthly patterns", pollutant=pollutant)
            
            if pollutant not in df.columns:
                raise DataProcessingError(f"Pollutant {pollutant} not found in data")
            
            # Extract month
            df_with_month = df.withColumn("month", month(col("datetime")))
            
            # Calculate monthly statistics
            monthly_stats = df_with_month.groupBy("month").agg(
                _mean(col(pollutant)).alias("mean"),
                _stddev(col(pollutant)).alias("std"),
                _min(col(pollutant)).alias("min"),
                _max(col(pollutant)).alias("max"),
                count(col(pollutant)).alias("count")
            ).orderBy("month")
            
            # Convert to pandas for analysis
            monthly_df = monthly_stats.toPandas()
            
            # Map month numbers to names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_df['month_name'] = monthly_df['month'].apply(lambda x: month_names[x - 1])
            
            # Find peak and low months
            peak_month = monthly_df.loc[monthly_df['mean'].idxmax()]
            low_month = monthly_df.loc[monthly_df['mean'].idxmin()]
            
            # Seasonal analysis
            spring_months = [3, 4, 5]  # Mar, Apr, May
            summer_months = [6, 7, 8]  # Jun, Jul, Aug
            fall_months = [9, 10, 11]  # Sep, Oct, Nov
            winter_months = [12, 1, 2]  # Dec, Jan, Feb
            
            spring_avg = monthly_df[monthly_df['month'].isin(spring_months)]['mean'].mean()
            summer_avg = monthly_df[monthly_df['month'].isin(summer_months)]['mean'].mean()
            fall_avg = monthly_df[monthly_df['month'].isin(fall_months)]['mean'].mean()
            winter_avg = monthly_df[monthly_df['month'].isin(winter_months)]['mean'].mean()
            
            result = {
                "pollutant": pollutant,
                "analysis_type": "monthly",
                "peak_month": {
                    "month": peak_month['month_name'],
                    "month_number": int(peak_month['month']),
                    "mean_value": float(peak_month['mean']),
                    "max_value": float(peak_month['max'])
                },
                "low_month": {
                    "month": low_month['month_name'],
                    "month_number": int(low_month['month']),
                    "mean_value": float(low_month['mean']),
                    "min_value": float(low_month['min'])
                },
                "seasonal_patterns": {
                    "spring_average": float(spring_avg) if not np.isnan(spring_avg) else None,
                    "summer_average": float(summer_avg) if not np.isnan(summer_avg) else None,
                    "fall_average": float(fall_avg) if not np.isnan(fall_avg) else None,
                    "winter_average": float(winter_avg) if not np.isnan(winter_avg) else None
                },
                "monthly_stats": monthly_df[['month_name', 'mean', 'std', 'min', 'max']].to_dict('records')
            }
            
            self.logger.info("Monthly pattern analysis completed",
                           peak_month=peak_month['month_name'],
                           low_month=low_month['month_name'])
            return result
            
        except Exception as e:
            self.logger.error("Monthly pattern analysis failed", error=str(e))
            raise DataProcessingError(f"Monthly pattern analysis failed: {e}")
    
    @log_performance
    def yearly_trends(self, df: DataFrame, pollutant: str) -> Dict[str, Any]:
        """Analyze yearly trends for a specific pollutant."""
        try:
            self.logger.info("Analyzing yearly trends", pollutant=pollutant)
            
            if pollutant not in df.columns:
                raise DataProcessingError(f"Pollutant {pollutant} not found in data")
            
            # Extract year
            df_with_year = df.withColumn("year", year(col("datetime")))
            
            # Calculate yearly statistics
            yearly_stats = df_with_year.groupBy("year").agg(
                _mean(col(pollutant)).alias("mean"),
                _stddev(col(pollutant)).alias("std"),
                _min(col(pollutant)).alias("min"),
                _max(col(pollutant)).alias("max"),
                count(col(pollutant)).alias("count")
            ).orderBy("year")
            
            # Convert to pandas for analysis
            yearly_df = yearly_stats.toPandas()
            
            if len(yearly_df) < 2:
                return {
                    "pollutant": pollutant,
                    "analysis_type": "yearly",
                    "message": "Insufficient data for yearly trend analysis",
                    "yearly_stats": yearly_df.to_dict('records')
                }
            
            # Calculate trend
            years = yearly_df['year'].values
            means = yearly_df['mean'].values
            
            # Simple linear regression for trend
            coeffs = np.polyfit(years, means, 1)
            trend_slope = coeffs[0]
            trend_intercept = coeffs[1]
            
            # Calculate trend significance (simple correlation)
            correlation = np.corrcoef(years, means)[0, 1]
            
            # Year-over-year changes
            yearly_df['yoy_change'] = yearly_df['mean'].pct_change() * 100
            yoy_avg_change = yearly_df['yoy_change'].mean()
            
            result = {
                "pollutant": pollutant,
                "analysis_type": "yearly",
                "trend_analysis": {
                    "slope": float(trend_slope),
                    "intercept": float(trend_intercept),
                    "correlation": float(correlation),
                    "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
                    "trend_strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                },
                "year_over_year": {
                    "average_percent_change": float(yoy_avg_change),
                    "changes": yearly_df[['year', 'yoy_change']].to_dict('records')
                },
                "yearly_stats": yearly_df[['year', 'mean', 'std', 'min', 'max']].to_dict('records')
            }
            
            self.logger.info("Yearly trend analysis completed",
                           trend_direction=result["trend_analysis"]["trend_direction"],
                           correlation=float(correlation))
            return result
            
        except Exception as e:
            self.logger.error("Yearly trend analysis failed", error=str(e))
            raise DataProcessingError(f"Yearly trend analysis failed: {e}")
    
    @log_performance
    def comprehensive_time_analysis(self, df: DataFrame, pollutants: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive time-based analysis for multiple pollutants."""
        try:
            self.logger.info("Starting comprehensive time analysis")
            
            if pollutants is None:
                pollutants = self.settings.pollutant_columns
            
            results = {}
            
            for pollutant in pollutants:
                if pollutant in df.columns:
                    self.logger.info("Analyzing pollutant", pollutant=pollutant)
                    
                    pollutant_analysis = {
                        "daily": self.daily_patterns(df, pollutant),
                        "weekly": self.weekly_patterns(df, pollutant),
                        "monthly": self.monthly_patterns(df, pollutant),
                        "yearly": self.yearly_trends(df, pollutant)
                    }
                    
                    results[pollutant] = pollutant_analysis
                else:
                    self.logger.warning("Pollutant not found in data", pollutant=pollutant)
            
            # Generate summary insights
            summary = self._generate_summary_insights(results)
            
            result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "pollutants_analyzed": list(results.keys()),
                "individual_analyses": results,
                "summary_insights": summary
            }
            
            self.logger.info("Comprehensive time analysis completed",
                           pollutants_count=len(results))
            return result
            
        except Exception as e:
            self.logger.error("Comprehensive time analysis failed", error=str(e))
            raise DataProcessingError(f"Comprehensive time analysis failed: {e}")
    
    def _generate_summary_insights(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary insights from time-based analyses."""
        try:
            insights = {
                "peak_pollution_times": {},
                "seasonal_patterns": {},
                "trends": {}
            }
            
            for pollutant, analysis in analyses.items():
                # Peak times
                daily_peak = analysis["daily"]["peak_hour"]["hour"]
                weekly_peak = analysis["weekly"]["peak_day"]["day"]
                monthly_peak = analysis["monthly"]["peak_month"]["month"]
                
                insights["peak_pollution_times"][pollutant] = {
                    "daily_peak_hour": daily_peak,
                    "weekly_peak_day": weekly_peak,
                    "monthly_peak_month": monthly_peak
                }
                
                # Seasonal patterns
                seasonal = analysis["monthly"]["seasonal_patterns"]
                max_season = max(seasonal, key=seasonal.get) if seasonal else None
                min_season = min(seasonal, key=seasonal.get) if seasonal else None
                
                insights["seasonal_patterns"][pollutant] = {
                    "highest_season": max_season,
                    "lowest_season": min_season,
                    "seasonal_data": seasonal
                }
                
                # Trends
                if "trend_analysis" in analysis["yearly"]:
                    trend = analysis["yearly"]["trend_analysis"]
                    insights["trends"][pollutant] = {
                        "direction": trend["trend_direction"],
                        "strength": trend["trend_strength"],
                        "correlation": trend["correlation"]
                    }
            
            return insights
            
        except Exception as e:
            self.logger.warning("Failed to generate summary insights", error=str(e))
            return {"error": str(e)}
