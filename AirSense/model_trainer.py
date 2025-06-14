import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

class ModelTrainer:
    """Handles machine learning model training for air quality prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_pollutant_columns(self, data):
        """Get available pollutant columns from data."""
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        return [col for col in pollutant_cols if col in data.columns]
    
    def prepare_features(self, data, target_pollutant):
        """Prepare features for model training."""
        # Select features
        feature_cols = []
        
        # Add other pollutants as features (excluding target)
        pollutant_cols = self.get_pollutant_columns(data)
        for col in pollutant_cols:
            if col != target_pollutant:
                feature_cols.append(col)
        
        # Add temporal features
        temporal_cols = ['hour', 'month', 'dayofweek', 'hour_sin', 'hour_cos', 
                        'month_sin', 'month_cos', 'day_sin', 'day_cos']
        for col in temporal_cols:
            if col in data.columns:
                feature_cols.append(col)
        
        # Add meteorological features if available
        meteo_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
        for col in meteo_cols:
            if col in data.columns:
                feature_cols.append(col)
        
        # Add lagged features
        for lag in [1, 2, 3, 6, 12, 24]:
            lag_col = f"{target_pollutant}_lag_{lag}"
            if lag < len(data):
                data[lag_col] = data[target_pollutant].shift(lag)
                feature_cols.append(lag_col)
        
        # Add rolling features
        for window in [3, 6, 12, 24]:
            if window < len(data):
                roll_col = f"{target_pollutant}_roll_{window}"
                data[roll_col] = data[target_pollutant].rolling(window=window).mean()
                feature_cols.append(roll_col)
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = data_clean[feature_cols]
        y = data_clean[target_pollutant]
        
        return X, y, feature_cols
    
    def train_model(self, data, target_pollutant, model_type, test_size=0.2, n_estimators=100):
        """Train a machine learning model."""
        try:
            # Prepare features
            X, y, feature_cols = self.prepare_features(data, target_pollutant)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[f"{model_type}_{target_pollutant}"] = scaler
            
            # Train model
            if model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = None
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{model_type}_{target_pollutant}.joblib")
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'metrics': metrics
            }, model_path)
            
            return model, metrics, feature_importance
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance."""
        if feature_importance is None or len(feature_importance) == 0:
            return None
        
        # Take top 15 features
        top_features = feature_importance.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def load_model(self, model_name):
        """Load a saved model."""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_saved_models(self):
        """Get list of saved models."""
        if not os.path.exists(self.models_dir):
            return []
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        return [f.replace('.joblib', '') for f in model_files]
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def plot_predictions(self, y_true, y_pred, title="Model Predictions"):
        """Plot actual vs predicted values."""
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500
        )
        
        return fig
    
    def cross_validate_model(self, data, target_pollutant, model_type, cv_folds=5):
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        try:
            # Prepare features
            X, y, _ = self.prepare_features(data, target_pollutant)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize model
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, 
                                      scoring='r2', n_jobs=-1)
            
            return {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
        except Exception as e:
            raise Exception(f"Error in cross-validation: {str(e)}")
