import pandas as pd
import numpy as np
from datetime import datetime

class HealthAlerts:
    """Handles health alert generation based on AQI standards."""
    
    def __init__(self):
        # EPA AQI breakpoints for different pollutants
        self.aqi_breakpoints = {
            'PM2.5': [
                (0, 12.0, 0, 50),      # Good
                (12.1, 35.4, 51, 100), # Moderate
                (35.5, 55.4, 101, 150), # Unhealthy for Sensitive Groups
                (55.5, 150.4, 151, 200), # Unhealthy
                (150.5, 250.4, 201, 300), # Very Unhealthy
                (250.5, 500.4, 301, 500)  # Hazardous
            ],
            'PM10': [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ],
            'NO2': [
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 2049, 301, 500)
            ],
            'SO2': [
                (0, 35, 0, 50),
                (36, 75, 51, 100),
                (76, 185, 101, 150),
                (186, 304, 151, 200),
                (305, 604, 201, 300),
                (605, 1004, 301, 500)
            ],
            'CO': [
                (0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 50.4, 301, 500)
            ],
            'O3': [
                (0, 54, 0, 50),
                (55, 70, 51, 100),
                (71, 85, 101, 150),
                (86, 105, 151, 200),
                (106, 200, 201, 300),
                (201, 604, 301, 500)
            ]
        }
        
        # AQI level descriptions
        self.aqi_levels = {
            (0, 50): {
                'level': 'Good',
                'color': '#00E400',
                'health_impact': 'Air quality is considered satisfactory, and air pollution poses little or no risk.',
                'recommendations': 'None required. Enjoy outdoor activities.'
            },
            (51, 100): {
                'level': 'Moderate',
                'color': '#FFFF00',
                'health_impact': 'Air quality is acceptable for most people. However, sensitive individuals may experience minor symptoms.',
                'recommendations': 'Unusually sensitive people should consider reducing prolonged outdoor exertion.'
            },
            (101, 150): {
                'level': 'Unhealthy for Sensitive Groups',
                'color': '#FF7E00',
                'health_impact': 'Members of sensitive groups may experience health effects. The general public is not likely to be affected.',
                'recommendations': 'Active children and adults, and people with respiratory disease should limit prolonged outdoor exertion.'
            },
            (151, 200): {
                'level': 'Unhealthy',
                'color': '#FF0000',
                'health_impact': 'Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.',
                'recommendations': 'Active children and adults, and people with respiratory disease should avoid prolonged outdoor exertion; everyone else should limit prolonged outdoor exertion.'
            },
            (201, 300): {
                'level': 'Very Unhealthy',
                'color': '#8F3F97',
                'health_impact': 'Health warnings of emergency conditions. The entire population is more likely to be affected.',
                'recommendations': 'Active children and adults, and people with respiratory disease should avoid all outdoor exertion; everyone else should limit outdoor exertion.'
            },
            (301, 500): {
                'level': 'Hazardous',
                'color': '#7E0023',
                'health_impact': 'Health alert: everyone may experience more serious health effects.',
                'recommendations': 'Everyone should avoid all outdoor exertion.'
            }
        }
    
    def calculate_aqi(self, pollutant, concentration):
        """Calculate AQI for a specific pollutant and concentration."""
        if pollutant not in self.aqi_breakpoints:
            return None
        
        breakpoints = self.aqi_breakpoints[pollutant]
        
        # Find the appropriate breakpoint
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Linear interpolation formula
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return round(aqi)
        
        # If concentration is above the highest breakpoint, return max AQI
        if concentration > breakpoints[-1][1]:
            return 500
        
        # If concentration is below the lowest breakpoint, return 0
        return 0
    
    def get_aqi_level_info(self, aqi):
        """Get AQI level information based on AQI value."""
        for (low, high), info in self.aqi_levels.items():
            if low <= aqi <= high:
                return info
        
        # Default for extreme values
        return self.aqi_levels[(301, 500)]
    
    def generate_alerts(self, forecast_data, target_pollutant):
        """Generate health alerts based on forecast data."""
        alerts = []
        
        for _, row in forecast_data.iterrows():
            concentration = row['forecast']
            timestamp = row['datetime']
            
            # Calculate AQI
            aqi = self.calculate_aqi(target_pollutant, concentration)
            
            if aqi is not None:
                # Get level information
                level_info = self.get_aqi_level_info(aqi)
                
                # Create alert
                alert = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'pollutant': target_pollutant,
                    'value': concentration,
                    'aqi': aqi,
                    'level': level_info['level'],
                    'color': level_info['color'],
                    'health_impact': level_info['health_impact'],
                    'recommendations': level_info['recommendations']
                }
                
                alerts.append(alert)
        
        return alerts
    
    def get_alert_summary(self, alerts):
        """Get summary statistics of alerts."""
        if not alerts:
            return {}
        
        # Count alerts by level
        level_counts = {}
        for alert in alerts:
            level = alert['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Find highest alert level
        aqi_values = [alert['aqi'] for alert in alerts]
        max_aqi = max(aqi_values)
        avg_aqi = sum(aqi_values) / len(aqi_values)
        
        # Time of highest alert
        max_alert = max(alerts, key=lambda x: x['aqi'])
        
        summary = {
            'total_alerts': len(alerts),
            'level_counts': level_counts,
            'max_aqi': max_aqi,
            'avg_aqi': round(avg_aqi, 1),
            'max_alert_time': max_alert['timestamp'],
            'max_alert_level': max_alert['level']
        }
        
        return summary
    
    def generate_health_recommendations(self, alert_level, sensitive_groups=None):
        """Generate detailed health recommendations based on alert level."""
        base_recommendations = self.get_aqi_level_info(
            {'Good': 25, 'Moderate': 75, 'Unhealthy for Sensitive Groups': 125,
             'Unhealthy': 175, 'Very Unhealthy': 250, 'Hazardous': 400}[alert_level]
        )['recommendations']
        
        detailed_recommendations = {
            'general_public': base_recommendations,
            'sensitive_groups': {},
            'outdoor_activities': {},
            'indoor_precautions': {}
        }
        
        # Specific recommendations for sensitive groups
        if alert_level in ['Good', 'Moderate']:
            detailed_recommendations['sensitive_groups'] = {
                'children': 'Normal outdoor activities are fine.',
                'elderly': 'Normal outdoor activities are fine.',
                'respiratory_conditions': 'Normal outdoor activities are fine.',
                'heart_conditions': 'Normal outdoor activities are fine.'
            }
            detailed_recommendations['outdoor_activities'] = {
                'exercise': 'Safe for all types of outdoor exercise.',
                'sports': 'All outdoor sports activities are safe.',
                'walking': 'Walking and jogging are safe.'
            }
            detailed_recommendations['indoor_precautions'] = {
                'windows': 'Safe to keep windows open for ventilation.',
                'air_purifiers': 'Not necessary unless preferred.',
                'masks': 'Not necessary for outdoor activities.'
            }
        
        elif alert_level == 'Unhealthy for Sensitive Groups':
            detailed_recommendations['sensitive_groups'] = {
                'children': 'Limit prolonged outdoor play, especially vigorous activities.',
                'elderly': 'Consider reducing time spent outdoors.',
                'respiratory_conditions': 'Limit outdoor activities, use inhaler as prescribed.',
                'heart_conditions': 'Avoid strenuous outdoor activities.'
            }
            detailed_recommendations['outdoor_activities'] = {
                'exercise': 'Sensitive individuals should exercise indoors.',
                'sports': 'Consider indoor alternatives for sensitive individuals.',
                'walking': 'Short walks are generally fine for most people.'
            }
            detailed_recommendations['indoor_precautions'] = {
                'windows': 'Consider keeping windows closed during peak pollution hours.',
                'air_purifiers': 'Consider using air purifiers indoors.',
                'masks': 'Sensitive individuals may benefit from masks outdoors.'
            }
        
        elif alert_level == 'Unhealthy':
            detailed_recommendations['sensitive_groups'] = {
                'children': 'Avoid outdoor activities, especially vigorous play.',
                'elderly': 'Stay indoors as much as possible.',
                'respiratory_conditions': 'Avoid all outdoor activities, monitor symptoms closely.',
                'heart_conditions': 'Stay indoors, avoid any strenuous activities.'
            }
            detailed_recommendations['outdoor_activities'] = {
                'exercise': 'Exercise indoors only.',
                'sports': 'Cancel outdoor sports activities.',
                'walking': 'Limit to essential outdoor activities only.'
            }
            detailed_recommendations['indoor_precautions'] = {
                'windows': 'Keep windows closed, use air conditioning if available.',
                'air_purifiers': 'Use air purifiers with HEPA filters.',
                'masks': 'Wear N95 masks when going outdoors is necessary.'
            }
        
        elif alert_level in ['Very Unhealthy', 'Hazardous']:
            detailed_recommendations['sensitive_groups'] = {
                'children': 'Stay indoors, avoid all outdoor exposure.',
                'elderly': 'Stay indoors, consider seeking medical advice if symptoms occur.',
                'respiratory_conditions': 'Stay indoors, have emergency medications ready.',
                'heart_conditions': 'Stay indoors, monitor symptoms, seek medical help if needed.'
            }
            detailed_recommendations['outdoor_activities'] = {
                'exercise': 'All exercise should be done indoors.',
                'sports': 'Cancel all outdoor activities.',
                'walking': 'Avoid all non-essential outdoor activities.'
            }
            detailed_recommendations['indoor_precautions'] = {
                'windows': 'Keep all windows closed, seal gaps if possible.',
                'air_purifiers': 'Use multiple air purifiers, ensure HEPA filtration.',
                'masks': 'Wear N95 or better masks for any outdoor exposure.'
            }
        
        return detailed_recommendations
    
    def create_alert_notification(self, alert):
        """Create a formatted alert notification."""
        notification = {
            'title': f"🚨 {alert['level']} Air Quality Alert",
            'subtitle': f"{alert['pollutant']} - AQI: {alert['aqi']}",
            'message': f"Concentration: {alert['value']:.1f} μg/m³ at {alert['timestamp']}",
            'health_impact': alert['health_impact'],
            'recommendations': alert['recommendations'],
            'color': alert['color'],
            'urgency': self._get_alert_urgency(alert['level'])
        }
        
        return notification
    
    def _get_alert_urgency(self, level):
        """Get urgency level for alert."""
        urgency_map = {
            'Good': 'low',
            'Moderate': 'low',
            'Unhealthy for Sensitive Groups': 'medium',
            'Unhealthy': 'high',
            'Very Unhealthy': 'critical',
            'Hazardous': 'emergency'
        }
        
        return urgency_map.get(level, 'medium')
    
    def export_alerts(self, alerts, filename=None):
        """Export alerts to CSV file."""
        if not alerts:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_alerts_{timestamp}.csv"
        
        # Convert alerts to DataFrame
        df = pd.DataFrame(alerts)
        df.to_csv(filename, index=False)
        
        return filename
