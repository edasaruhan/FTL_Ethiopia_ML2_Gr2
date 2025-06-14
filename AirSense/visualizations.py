import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class Visualizations:
    """Handles all visualization functions for air quality data."""
    
    def __init__(self):
        self.color_palette = [
            '#2E8B57', '#4682B4', '#DC143C', '#FFD700', 
            '#9932CC', '#FF6347', '#00CED1', '#FF69B4'
        ]
    
    def get_pollutant_columns(self, data):
        """Get available pollutant columns from data."""
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        return [col for col in pollutant_cols if col in data.columns]
    
    def plot_time_series(self, data, pollutants=None):
        """Plot time series for selected pollutants."""
        if pollutants is None:
            pollutants = self.get_pollutant_columns(data)
        
        fig = go.Figure()
        
        for i, pollutant in enumerate(pollutants):
            if pollutant in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['datetime'],
                    y=data[pollutant],
                    mode='lines',
                    name=pollutant,
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ))
        
        fig.update_layout(
            title='Air Quality Time Series',
            xaxis_title='Date',
            yaxis_title='Concentration (μg/m³)',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_correlation_matrix(self, data):
        """Plot correlation matrix of pollutants."""
        pollutant_cols = self.get_pollutant_columns(data)
        
        if len(pollutant_cols) < 2:
            # Return empty figure if not enough pollutants
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough pollutants for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate correlation matrix
        corr_matrix = data[pollutant_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            x=pollutant_cols,
            y=pollutant_cols,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Pollutant Correlation Matrix'
        )
        
        # Add correlation values as text
        for i in range(len(pollutant_cols)):
            for j in range(len(pollutant_cols)):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                )
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_distribution(self, data, pollutant):
        """Plot distribution of a specific pollutant."""
        if pollutant not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Pollutant {pollutant} not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Histogram',
                'Box Plot', 
                'Time Series',
                'Statistics'
            ),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[pollutant],
                name='Distribution',
                nbinsx=30,
                marker_color=self.color_palette[0]
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data[pollutant],
                name=pollutant,
                marker_color=self.color_palette[1]
            ),
            row=1, col=2
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(
                x=data['datetime'],
                y=data[pollutant],
                mode='lines',
                name='Time Series',
                line=dict(color=self.color_palette[2])
            ),
            row=2, col=1
        )
        
        # Statistics table
        stats = data[pollutant].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[
                    ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                    [f"{stats['count']:.0f}", f"{stats['mean']:.2f}", 
                     f"{stats['std']:.2f}", f"{stats['min']:.2f}",
                     f"{stats['25%']:.2f}", f"{stats['50%']:.2f}",
                     f"{stats['75%']:.2f}", f"{stats['max']:.2f}"]
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'{pollutant} Distribution Analysis',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_seasonal_patterns(self, data, pollutant):
        """Plot seasonal patterns for a pollutant."""
        if pollutant not in data.columns or 'month' not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Required data not available for seasonal analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate monthly averages
        monthly_avg = data.groupby('month')[pollutant].agg(['mean', 'std']).reset_index()
        monthly_avg['month_name'] = monthly_avg['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        fig = go.Figure()
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=monthly_avg['month_name'],
            y=monthly_avg['mean'],
            error_y=dict(type='data', array=monthly_avg['std']),
            mode='lines+markers',
            name=f'{pollutant} (±1 std)',
            line=dict(color=self.color_palette[0], width=3),
            marker=dict(size=8)
        ))
        
        # Add box plots for each month
        for month in range(1, 13):
            month_data = data[data['month'] == month][pollutant]
            if len(month_data) > 0:
                fig.add_trace(go.Box(
                    y=month_data,
                    name=monthly_avg[monthly_avg['month'] == month]['month_name'].iloc[0],
                    boxpoints='outliers',
                    marker_color=self.color_palette[1],
                    showlegend=False,
                    visible=False  # Hidden by default
                ))
        
        # Add buttons to toggle between line and box plots
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] + [False] * 12}],
                            label="Line Plot",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False] + [True] * 12}],
                            label="Box Plots",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        
        fig.update_layout(
            title=f'{pollutant} Seasonal Patterns',
            xaxis_title='Month',
            yaxis_title=f'{pollutant} Concentration (μg/m³)',
            height=500
        )
        
        return fig
    
    def plot_diurnal_patterns(self, data, pollutant):
        """Plot diurnal (daily) patterns for a pollutant."""
        if pollutant not in data.columns or 'hour' not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Required data not available for diurnal analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate hourly averages
        hourly_avg = data.groupby('hour')[pollutant].agg(['mean', 'std', 'median']).reset_index()
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color=self.color_palette[0], width=3),
            marker=dict(size=6)
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['median'],
            mode='lines',
            name='Median',
            line=dict(color=self.color_palette[1], width=2, dash='dash')
        ))
        
        # Add confidence band (mean ± std)
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'] + hourly_avg['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'] - hourly_avg['std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(46, 139, 87, 0.2)',
            fill='tonexty',
            name='±1 Standard Deviation',
            hoverinfo='skip'
        ))
        
        # Add vertical lines for key times
        fig.add_vline(x=6, line_dash="dot", line_color="gray", 
                     annotation_text="Morning Rush")
        fig.add_vline(x=18, line_dash="dot", line_color="gray", 
                     annotation_text="Evening Rush")
        
        fig.update_layout(
            title=f'{pollutant} Diurnal Patterns',
            xaxis_title='Hour of Day',
            yaxis_title=f'{pollutant} Concentration (μg/m³)',
            height=500,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2,
                range=[0, 23]
            )
        )
        
        return fig
    
    def plot_pollutant_comparison(self, data, pollutants=None):
        """Plot comparison of multiple pollutants over time."""
        if pollutants is None:
            pollutants = self.get_pollutant_columns(data)
        
        if len(pollutants) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="At least 2 pollutants required for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=len(pollutants), cols=1,
            shared_xaxes=True,
            subplot_titles=pollutants,
            vertical_spacing=0.05
        )
        
        for i, pollutant in enumerate(pollutants):
            if pollutant in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['datetime'],
                        y=data[pollutant],
                        mode='lines',
                        name=pollutant,
                        line=dict(color=self.color_palette[i % len(self.color_palette)])
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Pollutant Comparison',
            height=150 * len(pollutants),
            hovermode='x unified',
            showlegend=False
        )
        
        # Update y-axis titles
        for i, pollutant in enumerate(pollutants):
            fig.update_yaxes(title_text=f"{pollutant} (μg/m³)", row=i+1, col=1)
        
        fig.update_xaxes(title_text="Date", row=len(pollutants), col=1)
        
        return fig
    
    def plot_aqi_heatmap(self, data, pollutant):
        """Plot AQI heatmap showing patterns by hour and day."""
        if pollutant not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Pollutant {pollutant} not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create hour and day of week columns if not present
        if 'hour' not in data.columns:
            data['hour'] = data['datetime'].dt.hour
        if 'dayofweek' not in data.columns:
            data['dayofweek'] = data['datetime'].dt.dayofweek
        
        # Create pivot table
        heatmap_data = data.pivot_table(
            values=pollutant,
            index='hour',
            columns='dayofweek',
            aggfunc='mean'
        )
        
        # Day names
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.imshow(
            heatmap_data.T,
            x=list(range(24)),
            y=day_names,
            color_continuous_scale='Reds',
            aspect='auto',
            title=f'{pollutant} Concentration Heatmap (Day vs Hour)'
        )
        
        fig.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def create_dashboard_summary(self, data):
        """Create a summary dashboard with key metrics."""
        pollutants = self.get_pollutant_columns(data)
        
        if not pollutants:
            return None
        
        # Calculate summary statistics
        summary_stats = {}
        for pollutant in pollutants:
            summary_stats[pollutant] = {
                'current': data[pollutant].iloc[-1] if len(data) > 0 else 0,
                'avg_24h': data[pollutant].tail(24).mean() if len(data) >= 24 else data[pollutant].mean(),
                'max_24h': data[pollutant].tail(24).max() if len(data) >= 24 else data[pollutant].max(),
                'trend': self._calculate_trend(data[pollutant])
            }
        
        return summary_stats
    
    def _calculate_trend(self, series):
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return 'stable'
        
        recent = series.tail(12).mean()  # Last 12 hours
        previous = series.tail(24).head(12).mean()  # Previous 12 hours
        
        if len(series) < 24:
            recent = series.tail(len(series)//2).mean()
            previous = series.head(len(series)//2).mean()
        
        if recent > previous * 1.05:
            return 'increasing'
        elif recent < previous * 0.95:
            return 'decreasing'
        else:
            return 'stable'
