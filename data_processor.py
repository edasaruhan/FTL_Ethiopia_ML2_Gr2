import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Class to handle data loading and preprocessing for air quality data."""

    def __init__(self):
        # Define required columns based on typical Beijing dataset structure
        self.expected_columns = ["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        self.numeric_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        self.scaler = StandardScaler()

    def load_and_preprocess(self, file_path_or_df):
        """Loads data from CSV or uses provided DataFrame and preprocesses it."""
        try:
            if isinstance(file_path_or_df, str):
                df = pd.read_csv(file_path_or_df)
            elif isinstance(file_path_or_df, pd.DataFrame):
                df = file_path_or_df.copy()
            else:
                raise ValueError("Input must be a file path (string) or a pandas DataFrame.")

            # --- Core Preprocessing Steps ---
            df = self._process_datetime(df)
            df = self._handle_missing_values(df)
            df = self._remove_outliers(df)
            df = self._create_features(df)

            # Ensure numeric columns are float
            for col in self.numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors to NaN

            # Re-handle potential NaNs introduced by coercion or feature engineering
            df = self._handle_missing_values(df, final_pass=True)

            # Scaling (optional, can be done later before training)
            # df = self._scale_features(df)

            # Drop intermediate categorical columns if they exist
            df = df.drop(columns=["wind_category", "PM2.5_AQI_category"], errors='ignore')


            # Ensure all expected numeric columns exist, fill with 0 if not (after processing)
            for col in self.numeric_columns:
                 if col not in df.columns:
                     df[col] = 0

            return df

        except Exception as e:
            st.error(f"Error during data preprocessing: {str(e)}")
            # Optionally re-raise or return None/empty df
            raise e

    def _process_datetime(self, df):
        """Process datetime columns and set as index."""
        if all(col in df.columns for col in ["year", "month", "day", "hour"]):
            # Combine year, month, day, hour into a single datetime column
            df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
            # Drop original columns if needed, but keep them for now as they might be useful features initially
            # df = df.drop(["year", "month", "day", "hour"], axis=1)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        else:
            # Attempt to find a datetime column based on common names
            datetime_col = next((col for col in df.columns if any(pattern in col.lower() for pattern in ["date", "time")]), None)
            if datetime_col:
                df["datetime"] = pd.to_datetime(df[datetime_col])
                # df = df.drop(datetime_col, axis=1)
            else:
                # If no datetime info, raise error or create a dummy index
                raise ValueError("Could not find or create a datetime column.")

        df = df.set_index("datetime")
        df = df.sort_index()
        # Ensure hourly frequency, filling gaps if necessary
        df = df.asfreq(\'H\')
        return df

    def _handle_missing_values(self, df, final_pass=False):
        """Handle missing values using interpolation and fill."""
        numeric_cols_in_df = df.select_dtypes(include=np.number).columns

        for col in numeric_cols_in_df:
            if df[col].isnull().any():
                # Use linear interpolation first for time series
                df[col] = df[col].interpolate(method=\'linear\', limit_direction=\'both\', limit=6) # Limit gap to 6 hours
                # Forward fill remaining gaps
                df[col] = df[col].fillna(method=\'ffill\')
                # Backward fill any remaining gaps at the beginning
                df[col] = df[col].fillna(method=\'bfill\')
                # If it\'s the final pass and still NaNs, fill with 0 or median
                if final_pass and df[col].isnull().any():
                     df[col] = df[col].fillna(0) # Or df[col].median()

        # Handle potential categorical NaNs (though less common after processing)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown") # Fallback

        return df

    def _remove_outliers(self, df):
        """Cap outliers using the IQR method."""
        numeric_cols_in_df = df.select_dtypes(include=np.number).columns

        for col in numeric_cols_in_df:
            # Avoid capping engineered time features like hour, month etc.
            if col not in ["year", "month", "day", "hour", "day_of_week", "season",
                           "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
                           "wind_category_numeric", "PM2.5_AQI_numeric"]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR # Use 3*IQR for less aggressive capping
                upper_bound = Q3 + 3 * IQR

                # Cap values outside the bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def _create_features(self, df):
        """Create additional time-based and interaction features."""
        # Time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek # Monday=0, Sunday=6
        df["month"] = df.index.month
        df["day_of_year"] = df.index.dayofyear
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["quarter"] = df.index.quarter
        df["season"] = df.index.month % 12 // 3 + 1 # 1:Winter, 2:Spring, 3:Summer, 4:Fall

        # Cyclical encoding for time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Lag features for main pollutants
        pollutant_cols = [col for col in df.columns if col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]]
        for col in pollutant_cols:
            for lag in [1, 3, 6, 12, 24]: # 1hr to 24hr lags
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Rolling window features
        for col in pollutant_cols + ["TEMP", "WSPM"]:
             if col in df.columns:
                for window in [3, 6, 12, 24]: # 3hr to 24hr windows
                    df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()

        # Interaction features (example)
        if "TEMP" in df.columns and "WSPM" in df.columns:
            df["temp_wind_interaction"] = df["TEMP"] * df["WSPM"]

        # Wind speed categories (optional, keep numeric version)
        if "WSPM" in df.columns:
            df["wind_category"] = pd.cut(df["WSPM"], bins=[-np.inf, 2, 6, 12, np.inf], labels=["Calm", "Light", "Moderate", "Strong"])
            df["wind_category_numeric"] = df["wind_category"].cat.codes

        # PM2.5 AQI category (optional, keep numeric version)
        if "PM2.5" in df.columns:
            df["PM2.5_AQI_category"] = pd.cut(df["PM2.5"], bins=[0, 12, 35.4, 55.4, 150.4, 250.4, np.inf], labels=["Good", "Moderate", "USG", "Unhealthy", "Very_Unhealthy", "Hazardous"])
            df["PM2.5_AQI_numeric"] = df["PM2.5_AQI_category"].cat.codes

        return df

    def _scale_features(self, df):
        """Scale numeric features using StandardScaler."""
        # Select only numeric columns suitable for scaling (exclude identifiers, target, etc.)
        cols_to_scale = df.select_dtypes(include=np.number).columns
        # Be careful not to scale the target variable if it\'s present
        # Also exclude simple time features if cyclical ones are used
        cols_to_exclude = ["year", "month", "day", "hour", "day_of_week", "season", "quarter", "day_of_year", "week_of_year"]
        cols_to_scale = [col for col in cols_to_scale if col not in cols_to_exclude]

        if not cols_to_scale:
            return df

        df_scaled = self.scaler.fit_transform(df[cols_to_scale])
        df[cols_to_scale] = df_scaled
        return df

    def get_feature_columns(self, df, target_column):
        """Utility to get list of feature columns for modeling."""
        exclude_cols = [target_column] # Start with target
        # Add original time columns if cyclical are used
        exclude_cols.extend(["year", "month", "day", "hour", "day_of_week"])
        # Add categorical representations if numeric exist
        exclude_cols.extend(["wind_category", "PM2.5_AQI_category"])

        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ["int64", "float64", "int32", "float32"]]
        return feature_cols

# Example usage (for testing):
if __name__ == "__main__":
    # Create a dummy DataFrame similar to the expected input
    test_data = {
        "year": [2017, 2017, 2017, 2017, 2017],
        "month": [1, 1, 1, 1, 1],
        "day": [1, 1, 1, 1, 1],
        "hour": [0, 1, 2, 3, 4],
        "PM2.5": [50, 55, np.nan, 60, 500], # Include NaN and outlier
        "PM10": [70, 75, 80, 85, 90],
        "SO2": [10, 11, 12, 13, 14],
        "NO2": [30, 32, 34, 36, 38],
        "CO": [800, 850, 900, 950, 1000],
        "O3": [40, 38, 36, 34, 32],
        "TEMP": [0, 0.1, 0.2, 0.3, 0.4],
        "PRES": [1020, 1019.8, 1019.6, 1019.4, 1019.2],
        "DEWP": [-5, -4.9, -4.8, -4.7, -4.6],
        "RAIN": [0, 0, 0, 0, 0],
        "WSPM": [1, 1.1, 1.2, 1.3, 1.4]
    }
    test_df = pd.DataFrame(test_data)

    print("Original DataFrame:")
    print(test_df)

    processor = DataProcessor()
    try:
        processed_df = processor.load_and_preprocess(test_df)
        print("\nProcessed DataFrame:")
        print(processed_df.head())
        print("\nProcessed DataFrame Info:")
        processed_df.info()
        print("\nNaN check after processing:")
        print(processed_df.isnull().sum())
    except Exception as e:
        print(f"\nError during processing test: {e}")

