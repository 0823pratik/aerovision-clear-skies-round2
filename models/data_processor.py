# models/data_processor.py
"""
Optimized processor for your specific Google AirView+ Gurugram dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import TARGET_CITIES, DATA_DIR

class GurugramAirViewProcessor:
    def __init__(self):
        self.processed_data = None
        self.data_quality_report = {}
        
    def load_and_process_data(self, csv_path):
        """Load and process your Google AirView+ dataset"""
        print("🔄 Loading Google AirView+ Gurugram dataset...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df):,} records")
        
        # Filter for Gurugram data first
        df = self.filter_gurugram_data(df)
        
        # Data quality analysis
        self.analyze_data_quality(df)
        
        # Clean and process
        df_cleaned = self.clean_data(df)
        df_processed = self.create_features(df_cleaned)
        df_processed['AQI'] = self.calculate_aqi_safe(df_processed['PM2_5'])
        
        self.processed_data = df_processed
        
        print(f"✅ Processed {len(df_processed):,} records from {len(df_processed['station_name'].unique())} stations")
        return df_processed
    
    def filter_gurugram_data(self, df):
        """Filter data for Gurugram specifically"""
        print("🎯 Filtering for Gurugram data...")
        
        # Filter by city name
        gurugram_mask = df['city'].isin(TARGET_CITIES)
        gurugram_data = df[gurugram_mask].copy()
        
        # If no data found by city name, try geographic bounds
        if gurugram_data.empty:
            print("⚠️ No data found by city name, using geographic bounds...")
            
            # Gurugram geographic bounds
            lat_min, lat_max = 28.35, 28.55
            lon_min, lon_max = 76.90, 77.15
            
            geo_mask = (
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
            )
            gurugram_data = df[geo_mask].copy()
        
        print(f"📍 Found {len(gurugram_data):,} Gurugram records from {len(gurugram_data['station_name'].unique())} stations")
        return gurugram_data
    
    def get_latest_data(self, hours=1):
        """Get latest readings per station"""
        if self.processed_data is None:
            return pd.DataFrame()
        
        # Get most recent reading for each station
        latest_data = self.processed_data.groupby('station_name').last().reset_index()
        
        return latest_data

    def load_and_process(self, uploaded_file):
        """Wrapper method for uploaded files (Streamlit compatibility)"""
        return self.load_and_process_data(uploaded_file)

    def calculate_aqi(self, pm25_series):
        """Public method to calculate AQI"""
        return self.calculate_aqi_safe(pm25_series)

    def load_data_from_path(self, file_path=None):
        """Load data from specified path or default location"""
        if file_path is None:
            file_path = DATA_DIR / "raw" / "google_airview_data.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        
        return self.load_and_process_data(file_path)
    
    def analyze_data_quality(self, df):
        """Analyze data quality issues in your dataset"""
        print("🔍 Analyzing data quality...")
        
        # Check missing values
        missing_analysis = {
            'PM2_5': df['PM2_5'].isna().sum(),
            'PM10': df['PM10'].isna().sum(), 
            'CO2': df['CO2'].isna().sum(),
            'AT': df['AT'].isna().sum(),
            'RH': df['RH'].isna().sum()
        }
        
        self.data_quality_report = {
            'total_records': len(df),
            'missing_values': missing_analysis,
            'missing_percentages': {k: (v/len(df))*100 for k, v in missing_analysis.items()},
            'stations_with_issues': self.identify_problematic_stations(df)
        }
        
        print("📋 Data Quality Report:")
        for param, pct in self.data_quality_report['missing_percentages'].items():
            print(f"   {param}: {pct:.1f}% missing")
    
    def identify_problematic_stations(self, df):
        """Identify stations with frequent missing data"""
        station_quality = {}
        
        for station in df['station_name'].unique():
            if pd.isna(station):  # Skip NaN station names
                continue
                
            station_data = df[df['station_name'] == station]
            missing_pm25 = station_data['PM2_5'].isna().sum()
            total_records = len(station_data)
            missing_rate = (missing_pm25 / total_records) * 100 if total_records > 0 else 100
            
            station_quality[station] = {
                'missing_rate': missing_rate,
                'total_records': total_records,
                'status': 'problematic' if missing_rate > 20 else 'good'
            }
        
        return station_quality
    
    def clean_data(self, df):
        """Clean data handling missing values and outliers"""
        print("🧹 Cleaning data...")
        
        # Convert timestamp (handle your specific format)
        try:
            df['local_time'] = pd.to_datetime(df['local_time'], format='%d-%m-%Y %H:%M')
        except ValueError:
            # Try alternative formats
            try:
                df['local_time'] = pd.to_datetime(df['local_time'], format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                df['local_time'] = pd.to_datetime(df['local_time'], infer_datetime_format=True)
        
        # Handle missing values strategically
        df_clean = df.copy()
        
        # FIXED: For PM2.5 (most critical parameter) - Use modern pandas methods
        if df_clean['PM2_5'].isna().any():
            # Use forward fill within same station, then backward fill
            df_clean['PM2_5'] = (df_clean.groupby('station_name')['PM2_5']
                                .ffill()  # Modern method
                                .bfill()) # Modern method
            
            # If still missing, use station median
            station_medians = df_clean.groupby('station_name')['PM2_5'].median()
            for station in df_clean['station_name'].unique():
                if pd.isna(station):
                    continue
                mask = (df_clean['station_name'] == station) & df_clean['PM2_5'].isna()
                if mask.any() and not pd.isna(station_medians.get(station, np.nan)):
                    df_clean.loc[mask, 'PM2_5'] = station_medians[station]
        
        # For PM10 - estimate from PM2.5 if missing
        if df_clean['PM10'].isna().any():
            # Typical PM10/PM2.5 ratio in urban areas is ~1.5-2.0
            pm25_to_pm10_ratio = 1.7  # Conservative estimate
            
            missing_pm10_mask = df_clean['PM10'].isna() & df_clean['PM2_5'].notna()
            df_clean.loc[missing_pm10_mask, 'PM10'] = (
                df_clean.loc[missing_pm10_mask, 'PM2_5'] * pm25_to_pm10_ratio
            )
        
        # For CO2 - use station average or default outdoor air value
        if df_clean['CO2'].isna().any():
            station_co2_means = df_clean.groupby('station_name')['CO2'].mean()
            
            for station in df_clean['station_name'].unique():
                if pd.isna(station):
                    continue
                mask = (df_clean['station_name'] == station) & df_clean['CO2'].isna()
                if mask.any():
                    station_mean = station_co2_means.get(station, np.nan)
                    if not pd.isna(station_mean):
                        df_clean.loc[mask, 'CO2'] = station_mean
                    else:
                        df_clean.loc[mask, 'CO2'] = 400  # Default outdoor CO2 level
        
        # FIXED: For meteorological parameters (AT, RH) - Use modern pandas methods
        for param in ['AT', 'RH']:
            if df_clean[param].isna().any():
                # Use forward/backward fill within stations
                df_clean[param] = (df_clean.groupby('station_name')[param]
                                  .ffill()   # Modern method
                                  .bfill())  # Modern method
                
                # Use city-wide median as fallback
                city_median = df_clean[param].median()
                df_clean[param] = df_clean[param].fillna(city_median)
        
        # Remove unrealistic values
        df_clean = df_clean[
            (df_clean['PM2_5'] >= 0) & (df_clean['PM2_5'] <= 500) &
            (df_clean['PM10'] >= 0) & (df_clean['PM10'] <= 1000) &
            (df_clean['CO2'] >= 300) & (df_clean['CO2'] <= 5000) &
            (df_clean['AT'] >= 0) & (df_clean['AT'] <= 50) &
            (df_clean['RH'] >= 0) & (df_clean['RH'] <= 100)
        ]
        
        print(f"🎯 Data cleaned: {len(df_clean):,} valid records")
        return df_clean
    
    def create_features(self, df):
        """Create features for ML models"""
        print("⚙️ Creating features...")
        
        df_features = df.copy()
        
        # Temporal features
        df_features['hour'] = df_features['local_time'].dt.hour
        df_features['day_of_week'] = df_features['local_time'].dt.dayofweek
        df_features['month'] = df_features['local_time'].dt.month
        df_features['day_of_year'] = df_features['local_time'].dt.dayofyear
        
        # Time-based indicators
        df_features['is_rush_hour'] = df_features['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_night'] = df_features['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Air quality ratios (when both values are available)
        df_features['PM2_5_to_PM10_ratio'] = df_features['PM2_5'] / (df_features['PM10'] + 1)
        
        # Location-based features
        # Distance from Cyber Hub (major commercial center)
        cyber_hub_lat, cyber_hub_lon = 28.4949, 77.0869
        df_features['distance_from_cyber_hub'] = np.sqrt(
            (df_features['latitude'] - cyber_hub_lat)**2 + 
            (df_features['longitude'] - cyber_hub_lon)**2
        ) * 111  # Convert to km
        
        # Create location clusters based on coordinates
        try:
            from sklearn.cluster import KMeans
            if len(df_features) > 10:  # Need minimum data for clustering
                coords = df_features[['latitude', 'longitude']].values
                n_clusters = min(8, len(df_features['station_name'].unique()))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_features['location_cluster'] = kmeans.fit_predict(coords)
            else:
                df_features['location_cluster'] = 0
        except ImportError:
            print("⚠️ scikit-learn not available, skipping location clustering")
            df_features['location_cluster'] = 0
        
        # Station-based lag features (if enough temporal data)
        df_features = df_features.sort_values(['station_name', 'local_time'])
        
        # Simple lag features (1-hour lag)
        df_features['PM2_5_lag_1h'] = df_features.groupby('station_name')['PM2_5'].shift(1)
        
        # Additional useful features
        df_features['PM_total'] = df_features['PM2_5'] + df_features['PM10']
        df_features['temperature_humidity_index'] = df_features['AT'] / (df_features['RH'] + 1)
        
        return df_features
    
    def calculate_aqi_safe(self, pm25_series):
        """Calculate AQI with proper handling of missing values"""
        def pm25_to_aqi(pm25):
            if pd.isna(pm25):
                return np.nan
            
            # EPA AQI calculation for PM2.5
            if pm25 <= 12.0:
                return pm25 * 50 / 12.0
            elif pm25 <= 35.4:
                return 50 + (pm25 - 12.0) * 50 / (35.4 - 12.0)
            elif pm25 <= 55.4:
                return 100 + (pm25 - 35.4) * 50 / (55.4 - 35.4)
            elif pm25 <= 150.4:
                return 150 + (pm25 - 55.4) * 100 / (150.4 - 55.4)
            elif pm25 <= 250.4:
                return 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
            else:
                return 300 + (pm25 - 250.4) * 100 / (350.4 - 250.4)
        
        return pm25_series.apply(pm25_to_aqi)
    
    def get_station_reliability_score(self):
        """Calculate reliability score for each station"""
        if self.processed_data is None:
            return {}
        
        reliability_scores = {}
        
        for station in self.processed_data['station_name'].unique():
            if pd.isna(station):
                continue
                
            station_data = self.processed_data[self.processed_data['station_name'] == station]
            
            # Calculate various reliability metrics
            data_completeness = (station_data['PM2_5'].notna().sum() / len(station_data)) * 100
            temporal_consistency = len(station_data)  # More data points = more reliable
            
            # Value consistency (lower coefficient of variation = more consistent)
            pm25_mean = station_data['PM2_5'].mean()
            if pm25_mean > 0:
                cv = (station_data['PM2_5'].std() / pm25_mean) * 100
                value_consistency = max(0, 100 - cv)
            else:
                value_consistency = 0
            
            # Overall reliability score (0-100)
            reliability_score = (
                data_completeness * 0.6 + 
                min(temporal_consistency, 100) * 0.2 + 
                value_consistency * 0.2
            )
            
            reliability_scores[station] = {
                'score': min(100, max(0, reliability_score)),
                'data_completeness': data_completeness,
                'record_count': temporal_consistency,
                'value_consistency': value_consistency
            }
        
        return reliability_scores
    
    def get_current_status(self):
        """Get current air quality status across all stations"""
        if self.processed_data is None:
            return {
                'avg_pm25': 0,
                'max_pm25': 0,
                'avg_aqi': 0,
                'stations_reporting': 0,
                'hotspot_stations': [],
                'good_air_stations': []
            }
        
        latest_data = self.processed_data.groupby('station_name').last()
        
        return {
            'avg_pm25': latest_data['PM2_5'].mean(),
            'max_pm25': latest_data['PM2_5'].max(),
            'avg_aqi': latest_data['AQI'].mean(),
            'stations_reporting': len(latest_data),
            'hotspot_stations': latest_data[latest_data['PM2_5'] > 60].index.tolist(),
            'good_air_stations': latest_data[latest_data['PM2_5'] <= 35].index.tolist()
        }
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        if self.processed_data is None:
            return {}
        
        data = self.processed_data
        
        return {
            'total_records': len(data),
            'date_range': {
                'start': data['local_time'].min(),
                'end': data['local_time'].max()
            },
            'stations': {
                'total': len(data['station_name'].unique()),
                'list': data['station_name'].unique().tolist()
            },
            'pollution_stats': {
                'pm25_mean': data['PM2_5'].mean(),
                'pm25_max': data['PM2_5'].max(),
                'pm25_min': data['PM2_5'].min(),
                'aqi_mean': data['AQI'].mean(),
                'aqi_max': data['AQI'].max()
            },
            'data_quality': self.data_quality_report
        }
    
    def export_processed_data(self, output_path=None):
        """Export processed data to CSV"""
        if self.processed_data is None:
            raise ValueError("No processed data available to export")
        
        if output_path is None:
            output_path = DATA_DIR / "processed" / f"gurugram_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"✅ Processed data exported to: {output_path}")
        
        return output_path
