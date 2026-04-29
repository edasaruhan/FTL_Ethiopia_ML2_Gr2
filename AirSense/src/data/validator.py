"""Data validation utilities for AirSense."""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import ValidationError


class DataValidator:
    """Comprehensive data validation for air quality data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Validation thresholds
        self.thresholds = {
            "PM2.5": {"min": 0, "max": 500, "unit": "μg/m³"},
            "PM10": {"min": 0, "max": 600, "unit": "μg/m³"},
            "NO2": {"min": 0, "max": 200, "unit": "μg/m³"},
            "SO2": {"min": 0, "max": 1000, "unit": "μg/m³"},
            "CO": {"min": 0, "max": 50000, "unit": "μg/m³"},
            "O3": {"min": 0, "max": 300, "unit": "μg/m³"},
            "temperature": {"min": -50, "max": 60, "unit": "°C"},
            "humidity": {"min": 0, "max": 100, "unit": "%"},
            "pressure": {"min": 800, "max": 1200, "unit": "hPa"},
            "wind_speed": {"min": 0, "max": 100, "unit": "m/s"},
            "wind_direction": {"min": 0, "max": 360, "unit": "degrees"}
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame schema."""
        issues = []
        warnings = []
        
        # Check required columns
        required_columns = ["datetime"] + self.settings.pollutant_columns
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if "datetime" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                try:
                    pd.to_datetime(df["datetime"])
                except Exception as e:
                    issues.append(f"Invalid datetime format: {e}")
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        issues = []
        warnings = []
        quality_metrics = {}
        
        # Missing values analysis
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_analysis[col] = {
                "count": int(missing_count),
                "percentage": float(missing_percentage)
            }
            
            if missing_percentage > 50:
                issues.append(f"Column {col} has {missing_percentage:.1f}% missing values")
            elif missing_percentage > 20:
                warnings.append(f"Column {col} has {missing_percentage:.1f}% missing values")
        
        quality_metrics["missing_values"] = missing_analysis
        
        # Duplicate records
        duplicate_count = df.duplicated().sum()
        quality_metrics["duplicates"] = {
            "count": int(duplicate_count),
            "percentage": float((duplicate_count / len(df)) * 100)
        }
        
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate records")
        
        # Date range validation
        if "datetime" in df.columns:
            date_range = {
                "start": df["datetime"].min(),
                "end": df["datetime"].max(),
                "duration_days": (df["datetime"].max() - df["datetime"].min()).days
            }
            quality_metrics["date_range"] = date_range
            
            # Check for data gaps
            df_sorted = df.sort_values("datetime")
            time_gaps = df_sorted["datetime"].diff().dt.total_seconds() / 3600  # hours
            large_gaps = time_gaps[time_gaps > 24]  # gaps larger than 24 hours
            
            if len(large_gaps) > 0:
                warnings.append(f"Found {len(large_gaps)} gaps larger than 24 hours in data")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "metrics": quality_metrics
        }
    
    def validate_pollutant_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate pollutant value ranges."""
        issues = []
        warnings = []
        range_violations = {}
        
        for pollutant, threshold in self.thresholds.items():
            if pollutant not in df.columns:
                continue
                
            col_data = df[pollutant].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Check range violations
            below_min = (col_data < threshold["min"]).sum()
            above_max = (col_data > threshold["max"]).sum()
            
            if below_min > 0:
                issues.append(
                    f"{pollutant}: {below_min} values below minimum ({threshold['min']} {threshold['unit']})"
                )
            
            if above_max > 0:
                warnings.append(
                    f"{pollutant}: {above_max} values above maximum ({threshold['max']} {threshold['unit']})"
                )
            
            # Store violation details
            range_violations[pollutant] = {
                "below_min": int(below_min),
                "above_max": int(above_max),
                "threshold": threshold
            }
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "violations": range_violations
        }
    
    def validate_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal consistency of time series data."""
        issues = []
        warnings = []
        temporal_metrics = {}
        
        if "datetime" not in df.columns:
            return {"valid": False, "issues": ["No datetime column found"]}
        
        df_sorted = df.sort_values("datetime").copy()
        
        # Check for duplicate timestamps
        duplicate_timestamps = df_sorted["datetime"].duplicated().sum()
        if duplicate_timestamps > 0:
            issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Analyze time intervals
        time_diffs = df_sorted["datetime"].diff().dt.total_seconds() / 3600  # hours
        time_diffs = time_diffs.dropna()
        
        if len(time_diffs) > 0:
            temporal_metrics["time_intervals"] = {
                "min_hours": float(time_diffs.min()),
                "max_hours": float(time_diffs.max()),
                "median_hours": float(time_diffs.median()),
                "std_hours": float(time_diffs.std())
            }
            
            # Check for irregular intervals
            expected_interval = time_diffs.median()
            irregular_intervals = (time_diffs - expected_interval).abs() > (expected_interval * 0.5)
            
            if irregular_intervals.sum() > len(time_diffs) * 0.1:  # More than 10% irregular
                warnings.append(f"Highly irregular time intervals detected")
        
        # Check for data continuity
        if len(df_sorted) > 24:  # At least one day of hourly data
            expected_records = (df_sorted["datetime"].max() - df_sorted["datetime"].min()).total_seconds() / 3600 + 1
            actual_records = len(df_sorted)
            completeness = actual_records / expected_records
            
            temporal_metrics["completeness"] = {
                "expected_records": int(expected_records),
                "actual_records": actual_records,
                "completeness_ratio": float(completeness)
            }
            
            if completeness < 0.8:
                warnings.append(f"Data completeness is only {completeness:.1%}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "metrics": temporal_metrics
        }
    
    def comprehensive_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation on the dataset."""
        self.logger.info("Starting comprehensive data validation")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            }
        }
        
        # Run all validation checks
        validation_results["schema"] = self.validate_schema(df)
        validation_results["data_quality"] = self.validate_data_quality(df)
        validation_results["pollutant_ranges"] = self.validate_pollutant_ranges(df)
        validation_results["temporal_consistency"] = self.validate_temporal_consistency(df)
        
        # Overall validation status
        all_issues = []
        all_warnings = []
        
        for check in ["schema", "data_quality", "pollutant_ranges", "temporal_consistency"]:
            if check in validation_results:
                all_issues.extend(validation_results[check].get("issues", []))
                all_warnings.extend(validation_results[check].get("warnings", []))
        
        validation_results["overall"] = {
            "valid": len(all_issues) == 0,
            "issues_count": len(all_issues),
            "warnings_count": len(all_warnings),
            "issues": all_issues,
            "warnings": all_warnings
        }
        
        self.logger.info(
            "Validation completed",
            valid=validation_results["overall"]["valid"],
            issues_count=len(all_issues),
            warnings_count=len(all_warnings)
        )
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("AIRSENSE DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {validation_results['timestamp']}")
        report.append("")
        
        # Dataset info
        info = validation_results["dataset_info"]
        report.append("DATASET INFORMATION:")
        report.append(f"  Rows: {info['rows']:,}")
        report.append(f"  Columns: {info['columns']}")
        report.append(f"  Memory Usage: {info['memory_usage_mb']:.2f} MB")
        report.append("")
        
        # Overall status
        overall = validation_results["overall"]
        status = "✅ VALID" if overall["valid"] else "❌ INVALID"
        report.append(f"OVERALL STATUS: {status}")
        report.append(f"  Issues: {overall['issues_count']}")
        report.append(f"  Warnings: {overall['warnings_count']}")
        report.append("")
        
        # Issues
        if overall["issues"]:
            report.append("CRITICAL ISSUES:")
            for issue in overall["issues"]:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        # Warnings
        if overall["warnings"]:
            report.append("WARNINGS:")
            for warning in overall["warnings"]:
                report.append(f"  ⚠️  {warning}")
            report.append("")
        
        # Detailed metrics
        if "data_quality" in validation_results:
            dq = validation_results["data_quality"]["metrics"]
            if "missing_values" in dq:
                report.append("MISSING VALUES:")
                for col, stats in dq["missing_values"].items():
                    if stats["percentage"] > 0:
                        report.append(f"  {col}: {stats['count']} ({stats['percentage']:.1f}%)")
                report.append("")
        
        return "\n".join(report)


# Import required modules
import sys
