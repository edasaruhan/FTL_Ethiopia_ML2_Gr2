import pandas as pd
import numpy as np

class HealthAlertSystem:
    """Class to handle health alert generation based on air quality predictions."""

    def __init__(self):
        # Define AQI breakpoints and corresponding AQI values
        # Using EPA standards as a common reference
        # Format: (Concentration Low, Concentration High, AQI Low, AQI High)
        self.pm25_breakpoints = [
            (0.0, 12.0, 0, 50),      # Good
            (12.1, 35.4, 51, 100),     # Moderate
            (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups (USG)
            (55.5, 150.4, 151, 200),   # Unhealthy
            (150.5, 250.4, 201, 300),  # Very Unhealthy
            (250.5, 500.4, 301, 500)   # Hazardous (EPA tops out at 500)
            # Higher levels could be extrapolated if needed
        ]

        # NO2 breakpoints (1-hour standard, converted from ppb to 	μg/m³	 using factor ~1.88 at 25°C, 1 atm)
        # EPA standard: 1-hr NO2 101–360 ppb is USG, 361–649 Unhealthy, etc.
        self.no2_breakpoints = [
            (0, 100, 0, 50),         # Good (Using EPA 0-53 ppb ~ 0-100 	μg/m³	)
            (101, 188, 51, 100),       # Moderate (Using EPA 54-100 ppb ~ 101-188 	μg/m³	)
            (189, 677, 101, 150),      # USG (Using EPA 101-360 ppb ~ 189-677 	μg/m³	)
            (678, 1220, 151, 200),     # Unhealthy (Using EPA 361-649 ppb ~ 678-1220 	μg/m³	)
            (1221, 2349, 201, 300),    # Very Unhealthy (Using EPA 650-1249 ppb ~ 1221-2349 	μg/m³	)
            (2350, 4000, 301, 500)     # Hazardous (Using EPA 1250-2049 ppb ~ 2350-3853 	μg/m³	, capped)
        ]
        # Note: Other pollutants like SO2, CO, O3 have different breakpoints and averaging times.
        # This system currently focuses on PM2.5 and NO2 as per proposal.

        self.aqi_categories = {
            (0, 50): "Good",
            (51, 100): "Moderate",
            (101, 150): "Unhealthy for Sensitive Groups",
            (151, 200): "Unhealthy",
            (201, 300): "Very Unhealthy",
            (301, 500): "Hazardous"
            # AQI > 500 is also Hazardous
        }

        # Health recommendations (aligned with EPA AQI levels)
        self.health_recommendations = {
            "Good": {
                "General Public": "Air quality is excellent. Enjoy outdoor activities!",
                "Sensitive Groups": "Air quality poses little to no risk."
            },
            "Moderate": {
                "General Public": "Air quality is acceptable. Unusually sensitive individuals may experience minor symptoms.",
                "Sensitive Groups": "Consider reducing prolonged or heavy exertion outdoors if you experience symptoms."
            },
            "Unhealthy for Sensitive Groups": {
                "General Public": "Most people are unlikely to be affected. People with heart or lung disease, older adults, children, and teens should reduce prolonged or heavy exertion.",
                "Sensitive Groups": "Reduce prolonged or heavy exertion. Take more breaks during outdoor activities."
            },
            "Unhealthy": {
                "General Public": "Everyone may begin to experience health effects. People with heart or lung disease, older adults, children, and teens should avoid prolonged or heavy exertion.",
                "Sensitive Groups": "Avoid prolonged or heavy exertion. Consider moving activities indoors or rescheduling."
            },
            "Very Unhealthy": {
                "General Public": "Health alert: Everyone may experience more serious health effects. Everyone should avoid prolonged or heavy exertion.",
                "Sensitive Groups": "Avoid all physical activity outdoors. Move activities indoors or reschedule to a time with better air quality."
            },
            "Hazardous": {
                "General Public": "Health warning of emergency conditions: everyone is likely affected. Everyone should avoid all physical activity outdoors.",
                "Sensitive Groups": "Remain indoors and keep activity levels low. Follow advice from public health officials."
            },
            "Unknown": {
                "General Public": "Air quality data is unavailable.",
                "Sensitive Groups": "Air quality data is unavailable."
            }
        }

        self.alert_colors = {
            "Good": "#00E400", # Green
            "Moderate": "#FFFF00", # Yellow
            "Unhealthy for Sensitive Groups": "#FF7E00", # Orange
            "Unhealthy": "#FF0000", # Red
            "Very Unhealthy": "#8F3F97", # Purple
            "Hazardous": "#7E0023", # Maroon
            "Unknown": "#808080" # Grey
        }

    def _calculate_aqi(self, concentration, pollutant_type):
        """Calculates the AQI for a given concentration and pollutant type."""
        if pd.isna(concentration) or concentration < 0:
            return -1 # Indicate invalid input

        if pollutant_type.upper() == "PM2.5":
            breakpoints = self.pm25_breakpoints
        elif pollutant_type.upper() == "NO2":
            breakpoints = self.no2_breakpoints
        else:
            # Add breakpoints for other pollutants (SO2, CO, O3) if needed
            # For now, return -1 if pollutant is unsupported
            return -1

        for (conc_low, conc_high, aqi_low, aqi_high) in breakpoints:
            if conc_low <= concentration <= conc_high:
                # Linear interpolation formula
                aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
                return int(round(aqi))

        # If concentration exceeds the highest breakpoint
        if concentration > breakpoints[-1][1]:
             # Return max AQI or extrapolate if needed (EPA max is 500)
             return 500

        return -1 # Should not happen if breakpoints cover 0 upwards

    def get_aqi_category(self, aqi):
        """Determines the AQI category based on the AQI value."""
        if aqi < 0:
            return "Unknown"
        for (aqi_low, aqi_high), category in self.aqi_categories.items():
            if aqi_low <= aqi <= aqi_high:
                return category
        if aqi > 500:
             return "Hazardous" # AQI above 500 is still Hazardous
        return "Unknown"

    def generate_alerts(self, forecast_series, pollutant_type):
        """
        Generates health alerts for a forecast series.

        Args:
            forecast_series (pd.Series): Series with forecast values, indexed by datetime.
            pollutant_type (str): Type of pollutant (e.g., "PM2.5", "NO2").

        Returns:
            pd.DataFrame: DataFrame with timestamp, value, AQI, alert level, and recommendations.
        """
        alerts_data = []
        for timestamp, value in forecast_series.items():
            aqi = self._calculate_aqi(value, pollutant_type)
            alert_level = self.get_aqi_category(aqi)
            recommendations = self.get_health_recommendations(alert_level)

            alerts_data.append({
                "datetime": timestamp,
                "value": value,
                "pollutant": pollutant_type,
                "aqi": aqi if aqi >= 0 else np.nan, # Store AQI, use NaN for invalid
                "alert_level": alert_level,
                "recommendation_general": recommendations.get("General Public", "N/A"),
                "recommendation_sensitive": recommendations.get("Sensitive Groups", "N/A")
            })

        return pd.DataFrame(alerts_data).set_index("datetime")

    def get_health_recommendations(self, alert_level):
        """Gets health recommendations for a specific alert level."""
        return self.health_recommendations.get(alert_level, self.health_recommendations["Unknown"])

    def get_alert_color(self, alert_level):
        """Gets the color code for a specific alert level."""
        return self.alert_colors.get(alert_level, self.alert_colors["Unknown"])

    def generate_alert_summary(self, alerts_df):
        """
        Generates a summary of alerts for display.

        Args:
            alerts_df (pd.DataFrame): DataFrame generated by generate_alerts.

        Returns:
            dict: Summary statistics.
        """
        if alerts_df is None or alerts_df.empty:
            return {
                "total_hours": 0,
                "alert_distribution": {},
                "max_alert_level": "Unknown",
                "max_aqi": np.nan,
                "hours_unhealthy_or_worse": 0
            }

        distribution = alerts_df["alert_level"].value_counts().to_dict()
        max_aqi_row = alerts_df.loc[alerts_df["aqi"].idxmax()] if not alerts_df["aqi"].isnull().all() else None

        summary = {
            "total_hours": len(alerts_df),
            "alert_distribution": distribution,
            "max_alert_level": max_aqi_row["alert_level"] if max_aqi_row is not None else "Unknown",
            "max_aqi": max_aqi_row["aqi"] if max_aqi_row is not None else np.nan,
            "hours_unhealthy_or_worse": len(alerts_df[alerts_df["aqi"] >= 151]) # AQI >= 151 is Unhealthy
        }
        return summary

# Example usage (for testing):
if __name__ == "__main__":
    print("Testing Health Alert System...")
    health_system = HealthAlertSystem()

    # Test AQI calculation
    pm25_conc = 40.0
    no2_conc = 200.0
    aqi_pm25 = health_system._calculate_aqi(pm25_conc, "PM2.5")
    aqi_no2 = health_system._calculate_aqi(no2_conc, "NO2")
    print(f"PM2.5={pm25_conc} 	μg/m³	 => AQI={aqi_pm25} ({health_system.get_aqi_category(aqi_pm25)})")
    print(f"NO2={no2_conc} 	μg/m³	 => AQI={aqi_no2} ({health_system.get_aqi_category(aqi_no2)})")

    # Test alert generation
    forecast_dates = pd.date_range("2023-01-01 00:00", periods=5, freq="H")
    forecast_values = pd.Series([10, 25, 45, 60, 160], index=forecast_dates)
    alerts = health_system.generate_alerts(forecast_values, "PM2.5")
    print("\nGenerated Alerts:")
    print(alerts)

    # Test summary generation
    summary = health_system.generate_alert_summary(alerts)
    print("\nAlert Summary:")
    print(summary)

    # Test recommendations
    print("\nRecommendations for \'Unhealthy\':")
    print(health_system.get_health_recommendations("Unhealthy"))

    # Test colors
    print(f"\nColor for \'Moderate\': {health_system.get_alert_color(\"Moderate\")}")

