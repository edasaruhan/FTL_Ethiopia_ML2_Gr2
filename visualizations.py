import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class Visualizer:
    """Class to handle all visualization tasks for the AirSense dashboard."""

    def __init__(self):
        # Define a consistent color palette
        self.color_palette = px.colors.qualitative.Plotly
        self.pollutant_colors = {
            "PM2.5": "#FF6B6B", # Red
            "PM10": "#4ECDC4", # Teal
            "SO2": "#45B7D1", # Blue
            "NO2": "#FFA07A", # Light Salmon (distinct from PM2.5 red)
            "CO": "#FFD700", # Gold
            "O3": "#9370DB"  # Medium Purple
        }
        self.alert_colors = {
            "Good": "#00E400",
            "Moderate": "#FFFF00",
            "Unhealthy for Sensitive Groups": "#FF7E00",
            "Unhealthy": "#FF0000",
            "Very Unhealthy": "#8F3F97",
            "Hazardous": "#7E0023",
            "Unknown": "#808080"
        }

    def _get_pollutant_color(self, pollutant, index=0):
        """Helper to get color for a pollutant or fallback to palette."""
        return self.pollutant_colors.get(pollutant, self.color_palette[index % len(self.color_palette)])

    def create_time_series_plot(self, data, columns, title="Air Quality Time Series"):
        """Creates an interactive time series plot for selected columns."""
        fig = go.Figure()

        for i, col in enumerate(columns):
            if col in data.columns:
                color = self._get_pollutant_color(col, i)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode=\"lines\",
                    name=col,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<br><b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<extra></extra>"
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Date",
            yaxis_title="Concentration (units vary)", # Units depend on pollutant
            hovermode="x unified",
            legend_title_text="Pollutants",
            height=500,
            margin=dict(l=40, r=20, t=60, b=40)
        )

        # Add range selector for time series
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
        return fig

    def create_correlation_matrix(self, data, columns, title="Feature Correlation Matrix"):
        """Creates a correlation matrix heatmap for selected columns."""
        if len(columns) < 2:
            return self.create_placeholder_plot("Select at least two features for correlation analysis.")

        corr_matrix = data[columns].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r", # Reversed Red-Blue scale
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=600,
            width=600, # Keep it square
            xaxis_showgrid=False, yaxis_showgrid=False,
            yaxis_autorange=\"reversed\" # Show diagonal top-left to bottom-right
        )
        return fig

    def create_distribution_plot(self, data, column, title=None):
        """Creates a distribution plot (histogram + box plot)."""
        if column not in data.columns:
             return self.create_placeholder_plot(f"Column 	{column}	 not found in data.")

        if title is None:
            title = f"Distribution of {column}"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )

        color = self._get_pollutant_color(column)

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[column],
                name="Histogram",
                marker_color=color,
                opacity=0.75,
                hovertemplate=f"Range: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ),
            row=1, col=1
        )

        # Box plot
        fig.add_trace(
            go.Box(
                x=data[column], # Use x for horizontal box plot
                name="Box Plot",
                marker_color=color,
                boxpoints="outliers", # Show outliers
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate=f"Value: %{{x:.2f}}<extra></extra>"
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            height=500,
            bargap=0.01,
            margin=dict(l=40, r=20, t=60, b=40)
        )

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1) # Hide y-axis labels for box plot
        fig.update_xaxes(title_text=f"{column} Concentration", row=2, col=1)

        return fig

    def create_seasonal_analysis(self, data, column, title=None):
        """Creates a seasonal analysis plot (box plot by month)."""
        if column not in data.columns:
             return self.create_placeholder_plot(f"Column 	{column}	 not found.")

        if title is None:
            title = f"Monthly Patterns for {column}"

        monthly_data = data.copy()
        monthly_data["month_name"] = monthly_data.index.strftime("%B")
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        fig = go.Figure()
        color = self._get_pollutant_color(column)

        fig.add_trace(go.Box(
            y=monthly_data[column],
            x=monthly_data["month_name"],
            name="Monthly Distribution",
            marker_color=color
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Month",
            yaxis_title=f"{column} Concentration",
            xaxis=dict(categoryorder="array", categoryarray=month_order),
            boxmode="group",
            height=500,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    def create_daily_pattern_plot(self, data, column, title=None):
        """Creates a daily pattern plot (average by hour with std dev band)."""
        if column not in data.columns:
             return self.create_placeholder_plot(f"Column 	{column}	 not found.")

        if title is None:
            title = f"Average Daily Pattern for {column}"

        hourly_agg = data.groupby(data.index.hour)[column].agg(["mean", "std"]).reset_index()
        hourly_agg = hourly_agg.rename(columns={"datetime": "hour"})
        hourly_agg["std"] = hourly_agg["std"].fillna(0) # Fill NaN std dev if only one point

        fig = go.Figure()
        color = self._get_pollutant_color(column)
        rgb_color = px.colors.hex_to_rgb(color)
        fill_color = f"rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)"

        # Std deviation band
        fig.add_trace(go.Scatter(
            x=hourly_agg["hour"].tolist() + hourly_agg["hour"].tolist()[::-1],
            y=(hourly_agg["mean"] + hourly_agg["std"]).tolist() + (hourly_agg["mean"] - hourly_agg["std"]).clip(lower=0).tolist()[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Std Dev Band"
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=hourly_agg["hour"],
            y=hourly_agg["mean"],
            mode="lines+markers",
            name="Hourly Average",
            line=dict(color=color, width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>Hour</b>: %{{x}}:00<br><b>Avg {column}</b>: %{{y:.2f}}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Hour of Day",
            yaxis_title=f"Average {column} Concentration",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            yaxis=dict(rangemode="tozero"), # Ensure y-axis starts at 0
            hovermode="x unified",
            height=500,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    def create_model_comparison_plot(self, results):
        """Creates bar charts comparing model performance metrics."""
        if not results:
            return self.create_placeholder_plot("No model results available for comparison.")

        models = list(results.keys())
        metrics = ["rmse", "mae", "r2"]
        metric_names = {"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
        metric_colors = {"rmse": "#FF6B6B", "mae": "#4ECDC4", "r2": "#45B7D1"}

        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[metric_names[m] for m in metrics]
        )

        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, np.nan) for model in models]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric_names[metric],
                    marker_color=metric_colors[metric],
                    hovertemplate=f"<b>Model</b>: %{{x}}<br><b>{metric_names[metric]}</b>: %{{y:.3f}}<extra></extra>"
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title=dict(text="Model Performance Comparison", x=0.5),
            showlegend=False,
            height=400,
            bargap=0.2,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        # Lower R2 is better for RMSE/MAE, higher is better for R2
        fig.update_yaxes(title_text="Lower is Better", row=1, col=1)
        fig.update_yaxes(title_text="Lower is Better", row=1, col=2)
        fig.update_yaxes(title_text="Higher is Better", row=1, col=3)

        return fig

    def create_forecast_plot(self, historical_data, forecast_data, target_column):
        """Creates a plot showing historical data, forecast, and confidence intervals."""
        if forecast_data is None or forecast_data.empty:
            return self.create_placeholder_plot("No forecast data available to plot.")

        fig = go.Figure()
        hist_color = "#2E86C1" # Blue for historical
        forecast_color = "#E74C3C" # Red for forecast
        ci_color = "rgba(231, 76, 60, 0.2)" # Light red for CI

        # Historical data (show recent period, e.g., last 7 days)
        recent_historical = historical_data[target_column].iloc[-168:] # Last 7 days (168 hours)
        fig.add_trace(go.Scatter(
            x=recent_historical.index,
            y=recent_historical.values,
            mode="lines",
            name="Historical",
            line=dict(color=hist_color, width=2),
            hovertemplate=f"<b>Historical {target_column}</b>: %{{y:.2f}}<br><b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<extra></extra>"
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color=forecast_color, width=2, dash="dash"),
            hovertemplate=f"<b>Forecast {target_column}</b>: %{{y:.2f}}<br><b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<extra></extra>"
        ))

        # Confidence Interval band
        if "lower_bound" in forecast_data.columns and "upper_bound" in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                y=forecast_data["upper_bound"].tolist() + forecast_data["lower_bound"].tolist()[::-1],
                fill="toself",
                fillcolor=ci_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True, # Show legend item for CI
                name="Confidence Interval"
            ))

        fig.update_layout(
            title=dict(text=f"{target_column} Forecast vs Historical", x=0.5),
            xaxis_title="Date",
            yaxis_title=f"{target_column} Concentration",
            hovermode="x unified",
            height=500,
            legend_title_text="Data Type",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    def create_alert_timeline_plot(self, alerts_df, target_column):
        """Creates a timeline plot of pollutant values colored by alert level."""
        if alerts_df.empty:
            return self.create_placeholder_plot("No alert data available for timeline.")

        fig = px.scatter(
            alerts_df.reset_index(),
            x="datetime",
            y="value",
            color="alert_level",
            color_discrete_map=self.alert_colors,
            title="Predicted Air Quality Alert Levels",
            labels={"value": f"{target_column} Concentration", "datetime": "Time", "alert_level": "Alert Level"},
            hover_data={"alert_level": True, "value": ":.2f"}
        )

        # Add lines connecting points of the same color/level for better visualization
        for level in alerts_df["alert_level"].unique():
            df_level = alerts_df[alerts_df["alert_level"] == level]
            fig.add_trace(go.Scatter(
                x=df_level.index,
                y=df_level["value"],
                mode="lines",
                line_color=self.alert_colors.get(level, "#808080"),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig.update_layout(
            title=dict(x=0.5),
            height=400,
            hovermode="x unified",
            xaxis_title="Forecast Time",
            yaxis_title=f"{target_column} Concentration",
            legend_title_text="Alert Level",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        fig.update_traces(marker=dict(size=8))
        return fig

    def create_alert_distribution_pie(self, alert_summary):
        """Creates a pie chart showing the distribution of alert levels."""
        alert_dist = alert_summary.get("alert_distribution", {})
        if not alert_dist:
            return self.create_placeholder_plot("No alert distribution data.")

        fig = go.Figure(data=[go.Pie(
            labels=list(alert_dist.keys()),
            values=list(alert_dist.values()),
            marker_colors=[self.alert_colors.get(level, "#808080") for level in alert_dist.keys()],
            hole=.3 # Make it a donut chart
        )])

        fig.update_traces(textposition="inside", textinfo="percent+label", hoverinfo="label+percent+value")
        fig.update_layout(
            title=dict(text="Forecast Alert Level Distribution", x=0.5),
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        return fig

    def create_placeholder_plot(self, message):
        """Creates a placeholder plot with a text message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="grey")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300 # Smaller height for placeholders
        )
        return fig

# Example usage (for testing):
if __name__ == "__main__":
    # Create dummy data for testing plots
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    test_data = pd.DataFrame({
        "PM2.5": np.random.rand(100) * 50 + 10,
        "NO2": np.random.rand(100) * 30 + 5,
        "TEMP": np.random.rand(100) * 10 + 15
    }, index=dates)

    forecast_dates = pd.date_range(dates[-1] + pd.Timedelta(hours=1), periods=24, freq="H")
    forecast_data = pd.DataFrame({
        "forecast": np.random.rand(24) * 40 + 15,
        "lower_bound": np.random.rand(24) * 30 + 10,
        "upper_bound": np.random.rand(24) * 50 + 20
    }, index=forecast_dates)
    forecast_data["lower_bound"] = forecast_data[["forecast", "lower_bound"]].min(axis=1)
    forecast_data["upper_bound"] = forecast_data[["forecast", "upper_bound"]].max(axis=1)

    model_results = {
        "Random Forest": {"rmse": 10.5, "mae": 8.1, "r2": 0.75},
        "XGBoost": {"rmse": 9.8, "mae": 7.5, "r2": 0.78}
    }

    alert_levels = ["Good"]*5 + ["Moderate"]*10 + ["Unhealthy for Sensitive Groups"]*5 + ["Unhealthy"]*4
    alerts_df = pd.DataFrame({
        "value": forecast_data["forecast"],
        "alert_level": alert_levels
    }, index=forecast_dates)

    alert_summary = {"alert_distribution": pd.Series(alert_levels).value_counts().to_dict()}

    viz = Visualizer()

    print("Generating test plots...")
    # In a real streamlit app, you would use st.plotly_chart(fig)
    # Here we just create them
    fig1 = viz.create_time_series_plot(test_data, ["PM2.5", "NO2"])
    fig2 = viz.create_correlation_matrix(test_data, ["PM2.5", "NO2", "TEMP"])
    fig3 = viz.create_distribution_plot(test_data, "PM2.5")
    fig4 = viz.create_seasonal_analysis(test_data, "PM2.5")
    fig5 = viz.create_daily_pattern_plot(test_data, "NO2")
    fig6 = viz.create_model_comparison_plot(model_results)
    fig7 = viz.create_forecast_plot(test_data, forecast_data, "PM2.5")
    fig8 = viz.create_alert_timeline_plot(alerts_df, "PM2.5")
    fig9 = viz.create_alert_distribution_pie(alert_summary)
    fig10 = viz.create_placeholder_plot("This is a placeholder.")

    print("Test plots generated (not displayed). Check class methods.")

