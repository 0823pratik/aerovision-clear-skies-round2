"""
Enhanced mapping utilities for advanced visualization
"""
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

class EnhancedMapPlotter:
    def __init__(self):
        self.gurugram_center = [28.4595, 77.0266]
        
    def create_pollution_heatmap(self, data):
    # Enhanced validation
        required_columns = ['latitude', 'longitude', 'PM2_5', 'AQI', 'station_name']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if data.empty or missing_columns:
            print(f"⚠️ Missing required columns: {missing_columns}")
            return self.create_fallback_map()
        
        # Get latest data for each station
        latest_data = data.groupby('station_name').last().reset_index()
        
        # Create base map
        m = folium.Map(
            location=self.gurugram_center,
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add heat map layer
        heat_data = []
        for _, row in latest_data.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['PM2_5']):
                heat_data.append([
                    row['latitude'], 
                    row['longitude'], 
                    row['PM2_5'] / 100  # Normalize for heatmap
                ])
        
        if heat_data:
            HeatMap(heat_data, radius=20, blur=15, gradient = {
                    0.0: '#00ff00',  # Bright green
                    0.3: '#ffff00',  # Yellow  
                    0.6: '#ffa500',  # Orange
                    0.8: '#ff4500',  # Red-orange
                    1.0: '#8b0000'   # Dark red
                }).add_to(m)
        
        # Add station markers
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, row in latest_data.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Create popup content
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>{row['station_name']}</b><br>
                    <hr style="margin: 5px 0;">
                    <b>Air Quality Index:</b> {row['AQI']:.0f}<br>
                    <b>PM2.5:</b> {row['PM2_5']:.1f} μg/m³<br>
                    <b>PM10:</b> {row['PM10']:.1f} μg/m³<br>
                    <b>Temperature:</b> {row['AT']:.1f}°C<br>
                    <b>Humidity:</b> {row['RH']:.1f}%<br>
                    <b>CO2:</b> {row['CO2']:.0f} PPM<br>
                    <b>Last Updated:</b> {row['local_time']}<br>
                </div>
                """
                
                # Color based on AQI
                color = self.get_aqi_color(row['AQI'])
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=max(8, min(20, row['PM2_5'] / 5)),
                    popup=folium.Popup(popup_html, max_width=300),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(marker_cluster)
        
        # Add legend
        self.add_legend(m)
        
        return m._repr_html_()
    
    def get_aqi_color(self, aqi):
        """Get color based on AQI value"""
        if aqi <= 50:
            return 'green'
        elif aqi <= 100:
            return 'yellow'
        elif aqi <= 150:
            return 'orange'
        elif aqi <= 200:
            return 'red'
        else:
            return 'purple'
    
    def add_legend(self, map_obj):
        """Add AQI legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>AQI Legend</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Good (0-50)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Moderate (51-100)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> USG (101-150)</p>
        <p><i class="fa fa-circle" style="color:red"></i> Unhealthy (151+)</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def create_fallback_map(self):  # Keep this name consistent
        """Create fallback map when no data available"""
        m = folium.Map(location=self.gurugram_center, zoom_start=12)
        
        folium.Marker(
            self.gurugram_center,
            popup="Gurugram Center - No data available",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        return m._repr_html_()

