import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import streamlit as st
import joblib # For saving/loading models and scalers
import os

class AirQualityPredictor:
    """Class to handle ML model training, prediction, and evaluation for air quality."""

    def __init__(self, model_dir="models"):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.best_model_name = None
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_features_target(self, data, target_column):
        """Prepares features (X) and target (y) for model training."""
        # Use the get_feature_columns method from DataProcessor if available
        # Or redefine logic here based on processed data structure
        exclude_cols = [target_column]
        exclude_cols.extend(["year", "month", "day", "hour", "day_of_week"])
        exclude_cols.extend(["wind_category", "PM2.5_AQI_category"]) # Add non-numeric derived cols

        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ["int64", "float64", "int32", "float32"]]

        # Ensure no columns with excessive NaNs remain (should be handled by processor)
        feature_cols = [col for col in feature_cols if data[col].notna().all()]

        self.feature_names[target_column] = feature_cols

        X = data[feature_cols].copy()
        y = data[target_column].copy()

        # Ensure target has no NaNs
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

        if X.empty or y.empty:
             raise ValueError("No valid data left after preparing features and target. Check preprocessing.")

        return X, y

    def train_models(self, X, y, target_column, model_types, test_size=0.2, random_state=42, status_callback=None):
        """Trains specified ML models using time-series aware splitting and scaling."""
        results = {}
        current_progress = 30 # Initial progress after data prep

        # Time-aware train-test split
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        if X_train.empty or y_train.empty:
            raise ValueError("Training set is empty after split. Not enough data?")

        # Scale features
        if status_callback: status_callback(current_progress + 5, "Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_column] = scaler
        current_progress += 10

        model_step_progress = (90 - current_progress) / len(model_types) # Progress per model

        for model_type in model_types:
            model_start_progress = current_progress
            try:
                if status_callback: status_callback(int(model_start_progress), f"Training {model_type}...")

                if model_type == "Random Forest":
                    model = self._train_random_forest(X_train_scaled, y_train, random_state)
                elif model_type == "XGBoost":
                    model = self._train_xgboost(X_train_scaled, y_train, random_state)
                else:
                    st.warning(f"Model type 	{model_type}	 not recognized. Skipping.")
                    continue

                if status_callback: status_callback(int(model_start_progress + model_step_progress * 0.6), f"Evaluating {model_type}...")

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Cross-validation score (on training data)
                cv_scores = self._time_series_cv(model, X_train_scaled, y_train)

                # Feature importance
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_

                # Store model and results
                self.models[f"{target_column}_{model_type}"] = model
                results[model_type] = {
                    "model": model,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "predictions": y_pred,
                    "actuals": y_test,
                    "feature_importance": feature_importance,
                    "feature_names": self.feature_names[target_column]
                }
                if status_callback: status_callback(int(model_start_progress + model_step_progress), f"{model_type} trained successfully!")
                st.success(f"✅ {model_type} trained successfully!") # Keep success message

            except Exception as e:
                st.error(f"❌ Error training {model_type}: {str(e)}")
                if status_callback: status_callback(int(model_start_progress + model_step_progress), f"Error training {model_type}.")
                # Continue with the next model if one fails
            current_progress += model_step_progress

        # Select best model based on RMSE (for this target)
        if results:
            best_model_for_target = min(results.keys(), key=lambda k: results[k]["rmse"])
            self.best_model_name = f"{target_column}_{best_model_for_target}" # Store globally best? Or per target?
            # For now, let's assume we track the best overall if multiple targets were trained sequentially
            # Or maybe the app logic handles selecting the best for the *current* target

        # Save models and scaler
        self.save_model_components(target_column)

        return results

    def _train_random_forest(self, X_train, y_train, random_state):
        """Trains a RandomForestRegressor model."""
        model = RandomForestRegressor(
            n_estimators=100,       # Keep relatively low for speed
            max_depth=20,           # Limit depth
            min_samples_split=10,   # Increase min samples split
            min_samples_leaf=5,     # Increase min samples leaf
            random_state=random_state,
            n_jobs=-1               # Use all available cores
        )
        model.fit(X_train, y_train)
        return model

    def _train_xgboost(self, X_train, y_train, random_state):
        """Trains an XGBoost Regressor model."""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror" # Explicitly set objective
        )
        model.fit(X_train, y_train)
        return model

    def _time_series_cv(self, model, X, y, n_splits=5):
        """Performs time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Use neg_root_mean_squared_error for direct RMSE interpretation
        scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores # Return positive RMSE values

    def generate_forecast(self, historical_data, target_column, model_name_suffix, forecast_hours):
        """Generates a forecast for the specified time horizon using an iterative approach."""
        model_key = f"{target_column}_{model_name_suffix}"
        if model_key not in self.models:
            # Attempt to load if not in memory
            if not self.load_model_components(target_column):
                 raise ValueError(f"Model {model_key} not found and could not be loaded. Train the model first.")

        model = self.models[model_key]
        scaler = self.scalers[target_column]
        features_list = self.feature_names[target_column]

        # Use the last N hours of historical data needed for lags/rolling features
        # Max lag is 24, max rolling window is 24. Need at least 24 hours.
        required_history = 48 # Use more history to be safe
        recent_data = historical_data.iloc[-required_history:].copy()

        forecast_values = []
        last_timestamp = recent_data.index[-1]

        for i in range(forecast_hours):
            # Get the latest feature vector
            last_known_features = recent_data.iloc[-1][features_list].values.reshape(1, -1)

            # Scale the features
            last_known_features_scaled = scaler.transform(last_known_features)

            # Predict the next step
            next_prediction = model.predict(last_known_features_scaled)[0]
            forecast_values.append(next_prediction)

            # --- Update recent_data with the prediction for the next iteration --- 
            # This is crucial for multi-step forecasting but complex to do perfectly
            # We need to update the target value and recalculate lags/rolling features

            next_timestamp = last_timestamp + timedelta(hours=i + 1)
            new_row = pd.Series(index=recent_data.columns, dtype=\'float64\')
            new_row.name = next_timestamp
            new_row[target_column] = next_prediction

            # Fill other known future features (e.g., time features)
            new_row["hour"] = next_timestamp.hour
            new_row["day_of_week"] = next_timestamp.dayofweek
            new_row["month"] = next_timestamp.month
            # ... (add all other time features)
            new_row["hour_sin"] = np.sin(2 * np.pi * new_row["hour"] / 24)
            new_row["hour_cos"] = np.cos(2 * np.pi * new_row["hour"] / 24)
            # ... (add other cyclical features)

            # For unknown future features (weather, other pollutants), use persistence or simple forecast
            # Simplest: Carry forward the last known value
            for col in recent_data.columns:
                if pd.isna(new_row[col]) and col != target_column:
                    new_row[col] = recent_data[col].iloc[-1]

            # Append the new row (as DataFrame to concat)
            recent_data = pd.concat([recent_data, new_row.to_frame().T])

            # --- Recalculate lag and rolling features for the newly added row --- 
            # This is the most complex part and requires careful implementation
            # Simplified approach: Recalculate only for the last row
            last_row_index = recent_data.index[-1]
            for lag_col in features_list:
                 if "_lag_" in lag_col:
                     base_col, lag_str = lag_col.rsplit("_lag_", 1)
                     lag = int(lag_str)
                     if base_col in recent_data.columns and len(recent_data) > lag:
                         recent_data.loc[last_row_index, lag_col] = recent_data[base_col].iloc[-1-lag]
                 elif "_roll_" in lag_col:
                     # Recalculating rolling features iteratively is tricky
                     # For simplicity, we might accept some inaccuracy here or recompute on the fly
                     base_col, type_window = lag_col.split("_roll_")
                     stat, window_str = type_window.split("_")
                     window = int(window_str)
                     if base_col in recent_data.columns:
                         rolling_series = recent_data[base_col].rolling(window=window, min_periods=1)
                         if stat == "mean":
                              recent_data.loc[last_row_index, lag_col] = rolling_series.mean().iloc[-1]
                         elif stat == "std":
                              recent_data.loc[last_row_index, lag_col] = rolling_series.std().iloc[-1]
            # --- End of iterative update --- 

        # Create the final forecast DataFrame
        forecast_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_hours,
            freq=\'H\'
        )
        forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_timestamps)

        # Add simplified confidence intervals (using std dev of recent residuals or similar)
        # This is a placeholder; proper CIs require more advanced methods (e.g., bootstrapping, quantile regression)
        residual_std = np.std(forecast_values) * 0.1 # Very rough estimate
        forecast_df["lower_bound"] = forecast_df["forecast"] - 1.96 * residual_std
        forecast_df["upper_bound"] = forecast_df["forecast"] + 1.96 * residual_std
        # Ensure bounds are non-negative
        forecast_df[["lower_bound", "upper_bound"]] = forecast_df[["lower_bound", "upper_bound"]].clip(lower=0)

        return forecast_df

    def save_model_components(self, target_column):
        """Saves trained models and scaler for a target pollutant."""
        try:
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{target_column}.joblib")
            joblib.dump(self.scalers[target_column], scaler_path)

            # Save feature names
            features_path = os.path.join(self.model_dir, f"features_{target_column}.joblib")
            joblib.dump(self.feature_names[target_column], features_path)

            # Save models
            for model_key, model in self.models.items():
                if model_key.startswith(target_column):
                    model_path = os.path.join(self.model_dir, f"model_{model_key}.joblib")
                    joblib.dump(model, model_path)
            # print(f"Saved model components for {target_column}")
        except Exception as e:
            st.error(f"Error saving model components for {target_column}: {e}")

    def load_model_components(self, target_column):
        """Loads trained models and scaler for a target pollutant."""
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{target_column}.joblib")
            if os.path.exists(scaler_path):
                self.scalers[target_column] = joblib.load(scaler_path)
            else:
                return False # Cannot proceed without scaler

            # Load feature names
            features_path = os.path.join(self.model_dir, f"features_{target_column}.joblib")
            if os.path.exists(features_path):
                self.feature_names[target_column] = joblib.load(features_path)
            else:
                return False # Need feature names

            # Load models for this target
            loaded_models = False
            for model_type in ["Random Forest", "XGBoost"]: # Assuming these are the types used
                model_key = f"{target_column}_{model_type}"
                model_path = os.path.join(self.model_dir, f"model_{model_key}.joblib")
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    loaded_models = True

            # print(f"Loaded model components for {target_column}")
            return loaded_models
        except Exception as e:
            st.error(f"Error loading model components for {target_column}: {e}")
            return False

# Example usage (for testing):
if __name__ == "__main__":
    # Requires data_processor.py and a sample dataset
    from data_processor import DataProcessor
    from utils import load_sample_data

    print("Testing ML Models Module...")
    # Load and process sample data
    try:
        sample_df_raw = load_sample_data() # Assumes utils.py handles sample data loading
        processor = DataProcessor()
        processed_df = processor.load_and_preprocess(sample_df_raw)
        print("Sample data processed.")

        predictor = AirQualityPredictor(model_dir="test_models")
        target = "PM2.5"

        # Prepare features
        X, y = predictor.prepare_features_target(processed_df, target)
        print(f"Features ({X.shape}) and target ({y.shape}) prepared.")

        # Train models
        print("Training models...")
        results = predictor.train_models(X, y, target, ["Random Forest", "XGBoost"])
        print("Model training results:")
        for model_name, res in results.items():
            print(f"  {model_name}: RMSE={res["rmse"]:.2f}, R2={res["r2"]:.2f}")

        # Generate forecast
        if predictor.best_model_name:
            print(f"Generating forecast using best model: {predictor.best_model_name}...")
            # Extract model suffix from best_model_name
            model_suffix = predictor.best_model_name.split("_")[-1]
            forecast = predictor.generate_forecast(processed_df, target, model_suffix, forecast_hours=12)
            print("Forecast generated:")
            print(forecast)
        else:
            print("Skipping forecast generation as no best model was determined.")

        # Test loading
        print("Testing model loading...")
        predictor_loaded = AirQualityPredictor(model_dir="test_models")
        if predictor_loaded.load_model_components(target):
            print(f"Components for {target} loaded successfully.")
            # Try forecasting with loaded model
            if f"{target}_Random Forest" in predictor_loaded.models:
                 forecast_loaded = predictor_loaded.generate_forecast(processed_df, target, "Random Forest", forecast_hours=6)
                 print("Forecast with loaded RF model:")
                 print(forecast_loaded)
        else:
            print(f"Failed to load components for {target}.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

