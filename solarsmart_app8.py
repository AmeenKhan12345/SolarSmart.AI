# SolarSmart - AI-Driven Solar Panel Performance Forecasting Tool
# Updated with Enhanced Performance Analyzer & Digital Twin with Predictive Maintenance

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import json
import math
from datetime import datetime, time as dt_time, timedelta, date
import warnings
import pytz # Library to handle timezones
from geopy.geocoders import Nominatim
import time # Added for auto-refresh
import math # Added for simulator calculations
import joblib # NEW: For saving and loading the model
import os # NEW: To check for model file existence
from supabase import create_client, Client


warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SolarSmart - AI Solar Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-glass-card {
        background: rgba(255, 255, 255, 0.99);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 25px;
        color: #333;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    .chart-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
        color: #222;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
    }
    .sidebar-content {
        background-color: #f8f9fa;
    }
    .performance-good {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 5px 0;
    }
    .performance-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 5px 0;
    }
    .performance-danger {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 5px 0;
    }
    .chart-title {
        font-size: 1.5rem;
         font-weight: 600;
         margin-bottom: 15px;
         text-align: center;
         color: #222;
    }
</style>
""", unsafe_allow_html=True)

class WeatherAPI:
    @st.cache_data(ttl=3600) # Cache data for 1 hour
    def get_real_weather_forecast(location, forecast_days):
        """
        Fetches real weather forecast data from Open-Meteo API.
        1. Geocodes the location string to get latitude and longitude.
        2. Calls the Open-Meteo API for forecast data.
        3. Processes the data into a clean pandas DataFrame.
        """
        try:
            # 1. Geocode location to get coordinates
            geolocator = Nominatim(user_agent="solar_forecaster_app")
            location_data = geolocator.geocode(location)
            if location_data is None:
                st.error(f"Could not find coordinates for '{location}'. Please try a different location (e.g., 'Paris, France').")
                return None, None, None

            lat, lon = location_data.latitude, location_data.longitude

            # 2. Call Open-Meteo API
            api_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,relative_humidity_2m_mean,shortwave_radiation_sum,cloud_cover_mean",
                "forecast_days": forecast_days,
                "timezone": "auto"
            }

            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            # 3. Process data into a DataFrame
            daily_data = data['daily']
            df = pd.DataFrame()
            df['date'] = pd.to_datetime(daily_data['time'])
            df['temperature'] = daily_data['temperature_2m_max']

            # Convert daily radiation sum from MJ/m² to average W/m² to match the training data format.
            # Conversion: (MJ * 1,000,000) / 86,400 seconds_in_a_day
            df['irradiance'] = [(val * 1000000) / 86400 for val in daily_data['shortwave_radiation_sum']]

            df['humidity'] = daily_data['relative_humidity_2m_mean']
            df['cloud_cover'] = daily_data['cloud_cover_mean']

            return df, lat, lon

        except requests.exceptions.RequestException as e:
            st.error(f"API Request Error: {e}")
            return None, None, None
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            return None, None, None
            
    # --- NEW: Function to get historical weather data for model training ---
    @staticmethod
    @st.cache_data(ttl=86400) # Cache for one day
    def get_historical_weather(lat, lon, days=365):
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            api_url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,cloud_cover",
                "timezone": "auto"
            }
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data['hourly'])
            df = df.rename(columns={
                'time': 'date',
                'temperature_2m': 'temperature',
                'relative_humidity_2m': 'humidity',
                'shortwave_radiation': 'irradiance',
                'cloud_cover': 'cloud_cover'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
            return df
        except Exception as e:
            st.error(f"Failed to fetch historical weather data: {e}")
            return pd.DataFrame()

    # --- NEW: Function to get current weather for real-time prediction ---
    @staticmethod
    @st.cache_data(ttl=900) # Cache for 15 minutes
    def get_current_weather(lat, lon):
        try:
            api_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation",
                "timezone": "auto"
            }
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()['current']
            
            # Prepare a DataFrame in the same format as the training data
            current_weather = pd.DataFrame([{
                'temperature': data['temperature_2m'],
                'humidity': data['relative_humidity_2m'],
                'irradiance': data['shortwave_radiation'],
                'cloud_cover': data['cloud_cover']
            }])
            # --- FIX: Reorder the columns to match the training order ---
            feature_order = ['temperature', 'irradiance', 'humidity', 'cloud_cover']
            current_weather = current_weather[feature_order]

            return current_weather
        except Exception as e:
            st.error(f"Could not fetch current weather: {e}")
            return None


# --- Classes for data generation and prediction ---
class SolarDataGenerator:
    """Generates fake historical solar data for model training."""
    @staticmethod
    def generate_historical_data(num_days):
        dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='D'))
        data = {
            'date': dates,
            'temperature': np.random.uniform(5, 35, size=num_days),
            'irradiance': np.random.uniform(100, 1000, size=num_days), # Avg W/m^2
            'humidity': np.random.uniform(30, 90, size=num_days),
            'cloud_cover': np.random.uniform(0, 100, size=num_days)
        }
        df = pd.DataFrame(data)
        # Simple formula for "actual" output
        df['actual_output'] = (df['irradiance'] * (1 - df['cloud_cover'] / 150) * 0.15 * (1 - (df['temperature']-25)*0.004))
        df['actual_output'] = df['actual_output'].clip(lower=0)
        return df
    
    @staticmethod
    def generate_realistic_data(num_panels=10, days=30):
        """
        Generate more realistic data with complex environmental and fault models.
        - Realistic cloud cover (daily and hourly variations)
        - Panel soiling and random rain (cleaning) events
        - Persistent and intermittent panel faults
        - Improved temperature and humidity models
        """
        dates = pd.date_range(start='2025-01-01', periods=days, freq='D')
        data = []

        # Panel-specific characteristics
        panel_base_efficiency = np.random.normal(0.20, 0.015, num_panels)
        panel_degradation_rate = np.random.normal(0.5, 0.1, num_panels) / 100 / 365 # Daily degradation
        panel_soiling_factor = np.ones(num_panels)
        panel_health_status = np.ones(num_panels) # 1.0 is healthy

        # Day-by-day simulation
        for i, date in enumerate(dates):
            # --- Environmental Factors for the Day ---
            # 1. Seasonal factor for irradiance and temperature
            season_factor = 0.85 + 0.35 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365)

            # 2. Daily cloudiness level (0.0 = heavy clouds, 1.0 = clear sky)
            daily_cloud_factor = np.random.beta(a=5, b=2) * season_factor

            # 3. Daily base temperature
            base_temp = 18 + 12 * season_factor

            # 4. Rain event (cleans panels)
            if np.random.random() < 0.1: # 10% chance of rain
                panel_soiling_factor[:] = 1.0 # Rain washes panels clean

            # 5. Daily soiling accumulation
            panel_soiling_factor *= (1 - np.random.uniform(0.001, 0.003, num_panels))

            # --- Hourly Simulation (Daylight Hours) ---
            for hour in range(5, 20):
                # Sun's position in the sky
                hour_factor = max(0, np.sin(np.pi * (hour - 5) / 14))

                # Short-term cloud variability
                hourly_cloud_noise = max(0, 1 + np.random.normal(0, 0.2))
                current_cloud_factor = min(1, daily_cloud_factor * hourly_cloud_noise)

                # Final irradiance calculation
                base_irradiance = 1100 * hour_factor * current_cloud_factor
                irradiance = max(0, base_irradiance + np.random.normal(0, 20))

                # Final temperature & humidity
                temperature = base_temp + (15 * hour_factor * current_cloud_factor) + np.random.normal(0, 1.5)
                humidity = max(20, min(95, 80 - (temperature - 20) * 2 + np.random.normal(0, 5)))

                # --- Panel-by-Panel Simulation ---
                for panel_idx in range(num_panels):
                    # Check for new persistent faults
                    if panel_health_status[panel_idx] == 1.0 and np.random.random() < 0.0001: # 0.01% chance per hour
                        panel_health_status[panel_idx] = np.random.uniform(0.1, 0.5) # Permanent partial failure

                    # Calculate current efficiency for this specific panel
                    degradation = (1 - panel_degradation_rate[panel_idx]) ** i
                    current_efficiency = (panel_base_efficiency[panel_idx] *
                                          degradation *
                                          panel_soiling_factor[panel_idx] *
                                          panel_health_status[panel_idx])

                    # Check for intermittent (temporary) faults
                    if np.random.random() < 0.001: # 0.1% chance per hour
                        current_efficiency *= np.random.uniform(0.2, 0.7)

                    # Calculate panel output metrics
                    panel_area = 1.7 # m^2
                    energy_output = irradiance * current_efficiency * panel_area # In Watts
                    voltage = 24.0 + (temperature - 25) * -0.1 + np.random.normal(0, 0.5)
                    current = max(0, energy_output / voltage if voltage > 0 else 0)
                    power = voltage * current

                    data.append({
                        'datetime': date + pd.Timedelta(hours=hour, minutes=np.random.randint(0, 60)),
                        'panel_id': f'Panel_{panel_idx+1:02d}',
                        'irradiance': irradiance,
                        'temperature': temperature,
                        'humidity': humidity,
                        'energy_output': max(0, energy_output), # Wh per reading
                        'panel_voltage': voltage,
                        'panel_current': current,
                        'panel_power': max(0, power),
                        'ambient_temp': temperature - np.random.uniform(2, 5), # Ambient is usually cooler
                        'wind_speed': max(0, np.random.normal(10, 5))
                    })

        return pd.DataFrame(data)

class EnhancedAnomalyDetector:
    """Enhanced anomaly detection with multiple methods"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, data):
        """Detect anomalies using multiple features"""
        # Features for anomaly detection
        features = ['energy_output', 'panel_voltage', 'panel_current', 'panel_power']
        
        # Handle missing features
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            st.warning("Insufficient features for anomaly detection. Need at least energy_output and one other metric.")
            return data
        
        # Scale features
        feature_data = self.scaler.fit_transform(data[available_features].fillna(0))
        
        # Fit the detector
        self.detector.fit(feature_data)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        anomalies = self.detector.predict(feature_data)
        anomaly_scores = self.detector.score_samples(feature_data)
        
        # Add results to dataframe
        data_copy = data.copy()
        data_copy['anomaly'] = anomalies
        data_copy['anomaly_score'] = anomaly_scores
        data_copy['is_anomaly'] = anomalies == -1
        
        return data_copy
    
    def analyze_panel_health(self, data):
        """Analyze individual panel health"""
        panel_health = {}
        
        for panel_id in data['panel_id'].unique():
            panel_data = data[data['panel_id'] == panel_id]
            
            # Calculate health metrics
            anomaly_rate = (panel_data['is_anomaly'].sum() / len(panel_data)) * 100
            avg_output = panel_data['energy_output'].mean()
            output_std = panel_data['energy_output'].std()
            voltage_stability = panel_data['panel_voltage'].std()
            
            # Determine health status
            if anomaly_rate > 15:
                health_status = "Critical"
                priority = 1
            elif anomaly_rate > 8:
                health_status = "Poor"
                priority = 2
            elif anomaly_rate > 3:
                health_status = "Fair"
                priority = 3
            else:
                health_status = "Good"
                priority = 4
            
            panel_health[panel_id] = {
                'health_status': health_status,
                'anomaly_rate': anomaly_rate,
                'avg_output': avg_output,
                'output_stability': output_std,
                'voltage_stability': voltage_stability,
                'priority': priority,
                'total_readings': len(panel_data),
                'anomaly_count': panel_data['is_anomaly'].sum()
            }
        
        return panel_health

class SimpleSolarPredictor:
    """A simple machine learning model to predict solar output."""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['temperature', 'irradiance', 'humidity', 'cloud_cover']
        self.target = 'actual_output'

    def train(self, historical_data):
        X = historical_data[self.features]
        y = historical_data[self.target]
        
        # We train on 100% of the data for the best possible production model
        self.model.fit(X, y)

        return self.model

    def predict(self, weather_data):
        X_pred = weather_data[self.features]
        return self.model.predict(X_pred)

# --- NEW: DIGITAL TWIN AND PREDICTION FUNCTIONS ---
MODEL_FILE = 'solar_model.joblib'

# --- NEW: Function to load the pre-trained model ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

    
# --- NEW: Helper function for training the model ---
def train_and_save_model(location):
    """Fetches data, trains, and saves the model for a given location."""
    try:
        # 1. Geocode location to get coordinates
        geolocator = Nominatim(user_agent="solar_model_trainer")
        location_data = geolocator.geocode(location)
        if not location_data:
            st.error(f"Could not find coordinates for {location}.")
            return None, None, None
        
        lat, lon = location_data.latitude, location_data.longitude
        
        # 2. Fetch 1 year of historical weather data
        historical_weather = WeatherAPI.get_historical_weather(lat, lon)
        if historical_weather.empty:
            st.error("Failed to fetch historical data, cannot train model.")
            return None, None, None
        
        # 3. Create the target variable for the model to learn from
        df = historical_weather.copy()
        panel_area = 1.7
        panel_efficiency = 0.20
        temp_coeff = -0.004
        df['actual_output'] = (df['irradiance'] * panel_area * panel_efficiency * (1 + (df['temperature'] - 25) * temp_coeff))
        df.loc[df['irradiance'] < 50, 'actual_output'] = 0
        df['actual_output'] = df['actual_output'].clip(lower=0)
        
        # 4. Train the model
        predictor = SimpleSolarPredictor()
        trained_model = predictor.train(df)
        
        # 5. Save the newly trained model, overwriting the old one
        joblib.dump(trained_model, MODEL_FILE)
        
        # Return the details for the session state
        return location, lat, lon
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None, None



# --- 1. SET UP YOUR SUPABASE CONNECTION ---
# It's best practice to use st.secrets for these in a deployed app
SUPABASE_URL = ""
SUPABASE_KEY = ""
TABLE_NAME = "" # The name of your table in Supabase

# Initialize the Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error initializing Supabase client: {e}")
    # Stop the app if the client can't be created
    st.stop()

# --- 2. DATA FETCHING FUNCTION ---
@st.cache_data(ttl=5) # Cache for 5 seconds for a near real-time feel
def fetch_supabase_data(table_name, limit=200):
    """Fetches the last N rows of data from the Supabase table."""
    try:
        # Query the table, select all columns, order by timestamp descending, and limit results
        response = supabase.table(table_name).select("*").order("created_at", desc=True).limit(limit).execute()
        
        # The data is in response.data
        if response.data:
            df = pd.DataFrame(response.data)
            # Ensure timestamp is in the correct format and sort ascending for charts
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at').reset_index(drop=True)
            return df
        return pd.DataFrame() # Return empty if no data
        
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame()


def ai_twin_command_center_page():
    # --- Check for active city in session state and set a default ---
    if 'active_city' not in st.session_state:
        st.session_state.active_city = "Nagpur"
        st.session_state.active_lat = 21.1458
        st.session_state.active_lon = 79.0882
    
    active_city = st.session_state.active_city
    active_lat = st.session_state.active_lat
    active_lon = st.session_state.active_lon

    st.title(f"☀️ AI Twin Command Center: {active_city}")
    st.markdown("This dashboard uses the AI model trained for the active city to provide live predictions and forecasts.")

    # --- Important Warning about Sensor Location ---
    if active_city.lower() != "nagpur":
        st.warning(f"**Location Mismatch:** The AI model is now predicting for **{active_city}**. However, the 'Actual Power' feed is from a fixed sensor in **Nagpur**. The 'Performance Analysis' (Power Loss, Revenue Loss) is therefore a meaningless comparison.")

    DEMO_SCALING_FACTOR = 10/7047  # Example: 120mW / 7047mW = 0.017

    model = load_model()
    if model is None:
        st.error("No AI model found. Please go to the 'Performance Forecasting' page and generate a forecast to train a model.")
        return

    # Initialize session_state values for cache busting
    if 'cache_buster' not in st.session_state:
        st.session_state['cache_buster'] = None

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.session_state['cache_buster'] = time.time()
        st.rerun()

    
    
    
    latest_power_mw = 0.0
    latest_row = {}
    power_mw_series = pd.Series([0.0])

    # Check if credentials have been set
    if "YOUR_SUPABASE" in SUPABASE_URL or "your_sensor_data" in TABLE_NAME:
        st.warning("Please update the Supabase URL, Key, and Table Name in the code.")
        return

    # Fetch the data
    df = fetch_supabase_data(TABLE_NAME)

    if df.empty:
        st.warning("No data received from Supabase yet. Make sure your ESP32 is running and sending data.")
        return
    # Ensure your column is a datetime type with timezone (UTC)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    # Convert to Asia/Kolkata timezone
    df['timestamp_kolkata'] = df['created_at'].dt.tz_convert('Asia/Kolkata')


    # --- Data Processing ---
    # IMPORTANT: Adjust these column names to match your Supabase table
    power_mw_series = df['power'].fillna(0) 
    latest_data = df.iloc[-1]
    latest_power_mw = power_mw_series.iloc[-1]
    latest_power_mw = latest_power_mw * 1000
    
    # --- Real-time Prediction for the ACTIVE CITY ---
    predicted_power_mw = 0
    current_weather = WeatherAPI.get_current_weather(active_lat, active_lon)
    if current_weather is not None:
        predicted_power_w = model.predict(current_weather)[0]
        predicted_power_mw_raw = max(0, predicted_power_w * 1000)
    # --- NEW: APPLY THE SCALING FACTOR ---
    predicted_power_mw = predicted_power_mw_raw * DEMO_SCALING_FACTOR


    # --- Dashboard Layout ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Live Power Comparison")
        # Dynamic gauge range based on the SCALED prediction
        gauge_max = max(150, math.ceil(predicted_power_mw * 1.2))
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = round(latest_power_mw, 6),
            title = {'text': "Actual Power"},
            domain = {'row': 0, 'column': 0},
            gauge = {'axis': {'range': [0, gauge_max]}, 'bar': {'color': "gold"}}
        ))
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = round(predicted_power_mw, 2),
            title = {'text': f"AI Predicted Power ({active_city})"},
            domain = {'row': 0, 'column': 1},
            gauge = {'axis': {'range': [0, gauge_max]}, 'bar': {'color': "deepskyblue"}}
        ))
        fig.update_layout(
            grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
            height=250, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Live Metrics")
        voltage_val = latest_data.get('voltage')
        current_val = latest_data.get('current')
        current_energy = latest_data.get('energy')
        current_energy = current_energy * 1000
        current_val=current_val*1000
        st.metric("Voltage", f"{voltage_val:.2f} V" if pd.notna(voltage_val) else "N/A")
        st.metric("Current", f"{current_val:.1f} mA" if pd.notna(current_val) else "N/A")
        st.metric("Energy", f"{current_energy:.1f} mWh" if pd.notna(current_val) else "N/A")

    with col3:
        st.subheader("Environment")
        temp_val = latest_data.get('temperature')
        hum_val = latest_data.get('humidity')
        st.metric("Temperature", f"{temp_val:.1f} °C" if pd.notna(temp_val) else "N/A")
        st.metric("Humidity", f"{hum_val:.1f} %" if pd.notna(hum_val) else "N/A")
        





    st.markdown("---")
    
    st.subheader(f"Performance Analysis ({active_city} Prediction vs. Actual)")
    electricity_price_inr = st.number_input("Electricity Price (₹ per kWh)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)

    power_loss_mw = max(0, predicted_power_mw - latest_power_mw)
    power_loss_percent = (power_loss_mw / predicted_power_mw) * 100 if predicted_power_mw > 0 else 0
    
    if len(df) > 1:
        delta_hours = df['timestamp_kolkata'].diff().dt.total_seconds().fillna(0) / 3600.0
        actual_energy_mwh = (power_mw_series * delta_hours).sum()
    else:
        actual_energy_mwh = 0.0
    
    daylight_hours_so_far = max(0, (datetime.now() - datetime.now().replace(hour=6, minute=0)).total_seconds() / 3600)
    predicted_energy_mwh = predicted_power_mw * daylight_hours_so_far
    
    energy_loss_mwh = max(0, predicted_energy_mwh - actual_energy_mwh)
    revenue_loss_inr = (energy_loss_mwh / 1000 / 1000) * electricity_price_inr

    loss_col1, loss_col2, loss_col3 = st.columns(3)
    loss_col1.metric("⚡ Power Difference", f"{power_loss_mw:.2f} mW")
    loss_col2.metric("📉 % Difference", f"{power_loss_percent:.1f}%")
    loss_col3.metric("💸 Est. Daily Revenue Difference", f"₹ {revenue_loss_inr:.2f}")



    # --- REAL-TIME IMPACT ANALYSIS ---
    st.markdown("---")
    st.subheader("Real-Time Impact Analysis")

    col_impact1, col_impact2 = st.columns(2)

    # --- COLUMN 1: Based on the ACTUAL live data from your demo panel ---
    with col_impact1:
        st.markdown("##### Live Demo Panel Impact")
        
        # Calculate cumulative energy for the demo panel for today
        if len(df) > 1:
            power_mw_series = df['power'].fillna(0) * 1000 # Convert W to mW
            delta_hours = df['timestamp_kolkata'].diff().dt.total_seconds().fillna(0) / 3600.0
            actual_energy_mwh = (power_mw_series * delta_hours).sum()
        else:
            actual_energy_mwh = 0.0

        st.metric("⚡️ Energy Generated Today", f"{actual_energy_mwh:.2f} mWh")
        # --- NEW METRIC 1: Instantaneous Power ---
        # A standard LED uses about 20mW.
        leds_powered_now = latest_power_mw / 20.0
        st.metric("💡 Can Power Now (est.)", f"{leds_powered_now:.1f} LEDs")

        # Add a simple relatable metric for the small panel
        led_bulb_hours_demo = actual_energy_mwh / 900 # A 9W (9000mW) bulb
        st.metric("💡 Can power an LED bulb for", f"{led_bulb_hours_demo * 60:.1f} minutes")

        # --- NEW METRIC 2: Data Points ---
        data_points_today = len(df)
        st.metric("📊 Data Points Sent Today", f"{data_points_today}")


    # --- COLUMN 2: Based on the AI PREDICTED data for a full-scale system ---
    with col_impact2:
        st.markdown("##### Full-Scale System Potential (AI Prediction)")
        
        # Convert the instantaneous predicted power (mW) to other units
        predicted_power_w = predicted_power_mw_raw / 1000.0

        # Calculate predicted energy for the day so far
        daylight_hours_so_far = max(0, (datetime.now() - datetime.now().replace(hour=6, minute=0)).total_seconds() / 3600)
        predicted_energy_wh_today = predicted_power_w * daylight_hours_so_far

        # Relatable Metrics
        phones_charged_hourly = predicted_power_w / 15.0 # How many phones can be charged per hour
        ev_km_per_hour = (predicted_power_w / 1000) * 6 # Avg. EV gets ~6 km per kWh
        revenue_saved_per_day = (predicted_energy_wh_today / 1000) * electricity_price_inr # Use the price input
        co2_avoided_grams = (predicted_energy_wh_today / 1000) * 475.0

        st.metric("⚡️ Est. Energy Today", f"{predicted_energy_wh_today / 1000:.2f} kWh")
        st.metric("📱 Phones Charged (per hour)", f"{phones_charged_hourly:.1f}")
        st.metric("🚗 EV Range Added (per hour)", f"{ev_km_per_hour:.1f} km")
        st.metric("💨 CO₂ Avoided Today (est.)", f"{co2_avoided_grams:.1f} grams")
        st.metric("💸 Est. Revenue Saved Today", f"₹ {revenue_saved_per_day:.2f}")

    # (You can adjust daylight start/end as needed; here we assume 06:00-18:00 local)
    now = datetime.now()
    daylight_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    daylight_end   = now.replace(hour=18, minute=0, second=0, microsecond=0)
    total_daylight_hours = max(0, (daylight_end - daylight_start).total_seconds() / 3600.0)

    # hours into daylight so far
    if now < daylight_start:
        hours_elapsed = 0.0
    elif now > daylight_end:
        hours_elapsed = total_daylight_hours
    else:
        hours_elapsed = max(0.0, (now - daylight_start).total_seconds() / 3600.0)

    remaining_daylight_hours = max(0.0, total_daylight_hours - hours_elapsed)
    # Small explanation + tip
    st.info(
        "Displayed numbers are estimates. Measured values come directly from the sensor (hardware). "
        "Predicted values are AI-based projections (calibrated for the demo panel) and assume the current level "
        f"holds for the remaining daylight ({remaining_daylight_hours:.2f} h). For precise forecasting, use hourly weather profiles."
    )


    st.markdown("---")
    
    st.subheader(f"Live & Forecasted Trends for {active_city}")
    live_chart = go.Figure()
    live_chart.add_trace(go.Scatter(
        x=df['timestamp_kolkata'], y=power_mw_series,
        mode='lines', name='Actual Power (Nagpur)', line=dict(width=3, color='gold')
    ))
    live_chart.add_hline(y=predicted_power_mw, line_dash="dash", line_color="deepskyblue",
                       annotation_text=f"AI Prediction ({active_city})", annotation_position="bottom right")
                       
    live_chart.update_layout(
        transition_duration=500, xaxis_title="Time (IST)", yaxis_title="Power (mW)",
        yaxis_range=[0, 120], height=400
    )
    st.plotly_chart(live_chart, use_container_width=True)

    st.markdown("---")
    st.subheader(f"7-Day Power Forecast for {active_city}")
    if model:
        with st.spinner(f"Fetching weather forecast for {active_city}..."):
            forecast_weather, _, _ = WeatherAPI.get_real_weather_forecast(active_city, 7)
            if forecast_weather is not None and not forecast_weather.empty:
                daily_predictions_w = model.predict(forecast_weather[['temperature', 'irradiance', 'humidity', 'cloud_cover']])
                forecast_weather['predicted_power_mw'] = [max(0, p * 50) for p in daily_predictions_w]
                fig_forecast = px.bar(
                    forecast_weather, x='date', y='predicted_power_mw',
                    title=f"Predicted Average Daily Power for {active_city}",
                    labels={'date': 'Date', 'predicted_power_mw': 'Predicted Power (mW)'},
                    text='predicted_power_mw'
                )
                fig_forecast.update_traces(texttemplate='%{text:.1f} mW', textposition='outside', marker_color='deepskyblue')
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Could not retrieve weather forecast.")
    
    

    st.markdown("---")

    with st.expander("Show Raw Recent Readings"):
        # Ensure your column is a datetime type with timezone (UTC)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)

        # Convert to Asia/Kolkata timezone
        df['timestamp_kolkata'] = df['created_at'].dt.tz_convert('Asia/Kolkata')
        display_df = df[['timestamp_kolkata','voltage', 'current', 'power', 'energy', 'temperature', 'humidity', 'pressure']].tail(10).copy()
        st.dataframe(display_df, use_container_width=True)

    time.sleep(5)
    st.rerun()

   

# --- MAIN FUNCTION (UPDATED) ---
def main():
    # --- Initialize Session State ---
    if 'active_city' not in st.session_state:
        st.session_state.active_city = "Nagpur"
        st.session_state.active_lat = 21.1458
        st.session_state.active_lon = 79.0882
        # On first run, check if a default model exists.
        if not os.path.exists(MODEL_FILE):
            st.sidebar.warning("No default model found. Please go to 'Performance Forecasting' and generate a forecast for 'Nagpur' to begin.")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "📈 Performance Forecasting",
        "🛰️ AI Twin Command Center",
        "🔍 Enhanced Performance Analyzer",
        "🎯 Scenario Simulator",
        "📊 Data Upload"
    ])
    
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📈 Performance Forecasting":
        forecasting_page()
    elif page == "🛰️ AI Twin Command Center":
        ai_twin_command_center_page()
    elif page == "🔍 Enhanced Performance Analyzer":
        enhanced_efficiency_page()
    elif page == "🎯 Scenario Simulator":
        simulator_page()
    elif page == "📊 Data Upload":
        data_upload_page()
        
def dashboard_page():
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px 30px;
        margin: 20px 0;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
        text-align: center;
    ">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #222; margin: 0;">
            ⚡ Solar Performance Dashboard
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.header("Solar Performance Dashboard")
    
    # Generate sample data
    data = SolarDataGenerator.generate_realistic_data(15,30)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_energy = data['energy_output'].sum()
        st.metric("Total Energy (kWh)", f"{total_energy:.1f}", delta="12.5%")
    
    with col2:
        avg_efficiency = data['energy_output'].mean()
        st.metric("Avg Daily Output", f"{avg_efficiency:.1f} kWh", delta="5.2%")
    
    with col3:
        max_output = data['energy_output'].max()
        st.metric("Peak Output", f"{max_output:.1f} kWh", delta="8.1%")
    
    with col4:
        uptime = 98.5
        st.metric("System Uptime", f"{uptime:.1f}%", delta="0.5%")
    

    # Charts
    col1, col2= st.columns(2)
    
    with col1:
        st.subheader("Energy Output Over Time")
        daily_output = data.groupby(data['datetime'].dt.date)['energy_output'].sum().reset_index()
        fig = px.line(daily_output, x='datetime', y='energy_output', 
                     title="Daily Energy Production")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Panel Performance Heatmap")
        # Group data to get total daily energy per panel
        panel_performance = data.groupby(['panel_id', data['datetime'].dt.date])['energy_output'].sum().reset_index()
        # Convert energy from Wh to kWh for better readability
        panel_performance['energy_output_kwh'] = panel_performance['energy_output'] / 1000
        # Pivot the data to create the matrix for the heatmap
        pivot_data = panel_performance.pivot(
            index='panel_id', 
            columns='datetime', 
            values='energy_output_kwh' # Use the new kWh column
        )
        # Create a more visually appealing and informative heatmap
        fig = px.imshow(
            pivot_data,
            aspect="auto",
            title="Daily Panel Performance (kWh)",
            labels=dict(x="Date", y="Panel ID", color="Energy (kWh)"), # Add clear labels
            color_continuous_scale=px.colors.sequential.YlOrRd # Use an intuitive color scale
        )
        
        # Update x-axis to format dates nicely and avoid clutter
        fig.update_xaxes(tickformat="%b %d", nticks=10)
        st.plotly_chart(fig, use_container_width=True)
   

    #   Charts
    col1, col2 = st.columns(2)

    with col1:
        irradiance_val = int(data['irradiance'].mean()) if 'irradiance' in data.columns else 850

        st.markdown(f"""
        <div class="metric-glass-card">
            <div class="chart-title">🔧 System Health</div>
            <div style="text-align: center; font-size: 1.2rem; font-weight: 500; color: #4CAF50; margin-bottom: 10px;">
                97.2% 
            </div>
            <div style="text-align: center; font-size: 1.2rem; font-weight: 500; color: #4CAF50; margin-bottom: 10px;">
                System Efficiency
            </div>

        <div style="padding: 6px 0; display: flex; justify-content: space-between; font-size: 1.1rem;">
            <div>🌡️ <strong>Temperature</strong></div>
            <div>37.2°C</div>
        </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                <span class="metric-label">💨 Wind Speed</span>
                <span class="metric-value" style="font-size: 1.2rem;">3.2 km/h</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                <span class="metric-label">☀️ Irradiance</span>
                <span class="metric-value" style="font-size: 1.2rem; color: #FFA500;">{irradiance_val} W/m²</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-glass-card">
            <div class="chart-title">🌤️ Perfect Conditions</div>
            <div style="text-align: center; font-size: 1.2rem; font-weight: 500; color: #4CAF50; margin-bottom: 10px;">
                Optimal Solar Generation Expected
            </div>
            <div style="font-size: 1.05rem;">
                <p><strong>UV Index:</strong> <span style="color: #FFD700; font-weight: 600;">High</span></p>
                <p><strong>Cloud Cover:</strong> <span style="color: #4CAF50; font-weight: 600;">15%</span></p>
                <p><strong>Visibility:</strong> <span style="color: #4CAF50; font-weight: 600;">Excellent</span></p>
                <p><strong>Humidity:</strong> <span style="color: #4CAF50; font-weight: 600;">45%</span></p>    
            </div>
        </div>
    """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive System Map
    st.markdown("""
    <div class="metric-glass-card" style="margin-top: 20px; padding: 20px;">
    <div class="chart-title">Interactive Solar Farm Layout</div>
    """, unsafe_allow_html=True)
    # Create a mock solar farm layout
    # Generate panel positions (grid layout)
    rows, cols = 3, 5
    panel_positions = []
    panel_ids = []
    panel_status = []
    panel_output = []
    
    for i in range(rows):
        for j in range(cols):
            panel_ids.append(f'Panel_{i*cols + j + 1:02d}')
            panel_positions.append([j * 2, i * 2])  # 2m spacing
            
            # Simulate different panel statuses
            if np.random.random() < 0.1:  # 5% chance of issue
                panel_status.append('Warning')
                panel_output.append(np.random.uniform(2, 4))
            elif np.random.random() < 0.2:  # 2% chance of critical
                panel_status.append('Critical')
                panel_output.append(np.random.uniform(0.5, 2))
            else:
                panel_status.append('Normal')
                panel_output.append(np.random.uniform(4, 6))
    
    # Create the layout visualization
    layout_df = pd.DataFrame({
        'Panel_ID': panel_ids,
        'X': [pos[0] for pos in panel_positions],
        'Y': [pos[1] for pos in panel_positions],
        'Status': panel_status,
        'Output': panel_output
    })
    
    # Color mapping for status
    color_map = {'Normal': '#4CAF50', 'Warning': '#FFD700', 'Critical': '#F44336'}
    layout_df['Color'] = layout_df['Status'].map(color_map)
    
    fig = go.Figure()
    
    for status in ['Normal', 'Warning', 'Critical']:
        status_data = layout_df[layout_df['Status'] == status]
        if not status_data.empty:
            fig.add_trace(go.Scatter(
                x=status_data['X'],
                y=status_data['Y'],
                mode='markers+text',
                name=f'{status} ({len(status_data)})',
                marker=dict(
                    size=50,
                    color=color_map[status],
                    symbol='square',
                    line=dict(color='white', width=2)
                ),
                text=status_data['Panel_ID'],
                textposition='middle center',
                textfont=dict(color='white', size=10, weight='bold'),
                hovertemplate='<b>%{text}</b><br>' +
                            'Status: ' + status + '<br>' +
                            'Output: %{customdata:.1f} kWh<br>' +
                            '<extra></extra>',
                customdata=status_data['Output']
            ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Position X (meters)',
            range=[-1, 9]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title='Position Y (meters)',
            range=[-1, 5]
        ),
        height=400,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom Stats Row
    st.markdown("<br>", unsafe_allow_html=True)

def forecasting_page():
    st.title("Solar Energy Forecaster")
    st.markdown("Enter a city to see its forecast. **This will also retrain the AI model** for the 'AI Twin Command Center' page.")

    # Use session state to remember the last city, default to Nagpur
    location = st.text_input("Enter a City (e.g., 'Paris', 'Tokyo')", value=st.session_state.get('active_city', 'Nagpur'))
    forecast_days = st.slider("Forecast Days", 1, 16, 7)
    
    # These inputs are now just for scaling this page's forecast visualization
    panel_capacity = st.number_input("Panel Capacity (kW)", value=10.0, min_value=0.5, step=0.5)
    panel_efficiency = st.slider("Panel Efficiency (%)", 15, 25, 20)

    # New, clearer button text
    if st.button("Generate Forecast & Retrain AI Twin", type="primary"):
        # We get weather_data, lat, and lon from our updated WeatherAPI function
        weather_data, lat, lon = WeatherAPI.get_real_weather_forecast(location, forecast_days)

        if weather_data is not None:
            # --- NEW: Retrain the main AI model for the new location ---
            with st.spinner(f"Retraining AI model for {location}... This may take a minute."):
                new_city, new_lat, new_lon = train_and_save_model(location)
            
            if new_city:
                st.success(f"AI model successfully retrained for {new_city}!")
                # Store the new city details in the session state for the other pages
                st.session_state.active_city = new_city
                st.session_state.active_lat = new_lat
                st.session_state.active_lon = new_lon
                # Clear the cached model to force a reload of the new file
                load_model.clear()
                st.info("Navigate to the 'AI Twin Command Center' to see the live predictions for your new city.")

            # --- Display the forecast on this page using the newly trained model ---
            st.markdown("---")
            st.subheader(f"Displaying Forecast for {location}")
            
            model = load_model() # Load the model we just saved
            if model:
                predictor = SimpleSolarPredictor()
                predictor.model = model
                
                with st.spinner(f"Generating forecast display for {location}..."):
                    predictions = predictor.predict(weather_data)
                    # Scale the raw model output based on user's panel specifications for this page's display
                    weather_data['predicted_output'] = (predictions * panel_capacity / 10 * panel_efficiency / 20)

                # --- THIS IS THE ORIGINAL CHARTING AND DISPLAY LOGIC ---
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Weather Forecast")
                    fig = make_subplots(rows=2, cols=2,
                                      subplot_titles=['Temperature (°C)', 'Irradiance (Avg W/m²)', 'Humidity (%)', 'Cloud Cover (%)'])

                    fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['temperature'], name='Temperature', line=dict(color='orangered')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['irradiance'], name='Irradiance', line=dict(color='gold')), row=1, col=2)
                    fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['humidity'], name='Humidity', line=dict(color='deepskyblue')), row=2, col=1)
                    fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['cloud_cover'], name='Cloud Cover', line=dict(color='darkgrey')), row=2, col=2)

                    fig.update_layout(height=500, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Energy Output Forecast (kWh)")
                    fig = px.bar(weather_data, x='date', y='predicted_output',
                                 title=f"Predicted Daily Energy Output",
                                 labels={'predicted_output': 'Energy (kWh)', 'date': 'Date'})
                    fig.update_traces(marker_color='#FFD700')
                    st.plotly_chart(fig, use_container_width=True)

                    total_predicted = weather_data['predicted_output'].sum()
                    avg_daily = weather_data['predicted_output'].mean()
                    peak_day = weather_data.loc[weather_data['predicted_output'].idxmax()]

                    st.markdown("#### Forecast Summary")
                    c1, c2 = st.columns(2)
                    c1.metric(f"Total Forecast ({forecast_days} days)", f"{total_predicted:.1f} kWh")
                    c2.metric("Average Daily Output", f"{avg_daily:.1f} kWh")
                    st.info(f"Peak production expected on **{peak_day['date'].strftime('%A, %b %d')}** with **{peak_day['predicted_output']:.1f} kWh**.")
            else:
                st.error("Could not load the newly trained model to generate the forecast display.")


def enhanced_efficiency_page():
    st.header("Enhanced Panel Performance Analyzer")
    
    # Data source selection
    data_source = st.radio("Select Data Source:", 
                          ["Generate Sample Data", "Manual Input", "Upload CSV"])
    
    data = None
    
    if data_source == "Generate Sample Data":
        st.subheader("Sample Data Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            num_panels = st.slider("Number of Panels", 5, 20, 10)
            days = st.slider("Days of Data", 7, 60, 30)
        
        with col2:
            fault_probability = st.slider("Fault Probability (%)", 0, 10, 2) / 100
            
        if st.button("Generate Data"):
            data = SolarDataGenerator.generate_realistic_data(num_panels, days)
            st.session_state['analysis_data'] = data
            st.success(f"Generated data for {num_panels} panels over {days} days")
    
    elif data_source == "Manual Input":
        st.subheader("Manual Data Entry")
        
        with st.expander("Add Manual Data Points"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                panel_id = st.text_input("Panel ID", value="Panel_01")
                energy_output = st.number_input("Energy Output (kWh)", value=5.0, step=0.1)
                panel_voltage = st.number_input("Panel Voltage (V)", value=24.0, step=0.1)
            
            with col2:
                panel_current = st.number_input("Panel Current (A)", value=2.5, step=0.1)
                temperature = st.number_input("Temperature (°C)", value=25.0, step=0.5)
                irradiance = st.number_input("Irradiance (W/m²)", value=800, step=10)
            
            with col3:
                humidity = st.number_input("Humidity (%)", value=60.0, step=1.0)
                wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.5)
                
            if st.button("Add Data Point"):
                new_data = {
                    'datetime': [datetime.datetime.now()],
                    'panel_id': [panel_id],
                    'energy_output': [energy_output],
                    'panel_voltage': [panel_voltage],
                    'panel_current': [panel_current],
                    'panel_power': [panel_voltage * panel_current],
                    'temperature': [temperature],
                    'irradiance': [irradiance],
                    'humidity': [humidity],
                    'wind_speed': [wind_speed],
                    'ambient_temp': [temperature + np.random.normal(0, 1)]
                }
                
                if 'manual_data' not in st.session_state:
                    st.session_state['manual_data'] = pd.DataFrame(new_data)
                else:
                    st.session_state['manual_data'] = pd.concat([
                        st.session_state['manual_data'], 
                        pd.DataFrame(new_data)
                    ], ignore_index=True)
                
                st.success("Data point added!")
        
        if 'manual_data' in st.session_state and not st.session_state['manual_data'].empty:
            st.subheader("Current Manual Data")
            st.dataframe(st.session_state['manual_data'])
            
            if st.button("Analyze Manual Data"):
                data = st.session_state['manual_data']
                st.session_state['analysis_data'] = data
    
    elif data_source == "Upload CSV":
        st.subheader("CSV File Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="performance_csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Column mapping
                st.subheader("Column Mapping")
                col1, col2, col3 = st.columns(3)
                
                required_columns = ['panel_id', 'energy_output', 'panel_voltage', 'panel_current']
                column_mapping = {}
                
                with col1:
                    column_mapping['panel_id'] = st.selectbox("Panel ID Column", data.columns)
                    column_mapping['energy_output'] = st.selectbox("Energy Output Column", data.columns)
                
                with col2:
                    column_mapping['panel_voltage'] = st.selectbox("Panel Voltage Column", data.columns)
                    column_mapping['panel_current'] = st.selectbox("Panel Current Column", data.columns)
                
                with col3:
                    column_mapping['datetime'] = st.selectbox("DateTime Column (optional)", 
                                                            ['None'] + list(data.columns))
                    column_mapping['temperature'] = st.selectbox("Temperature Column (optional)", 
                                                               ['None'] + list(data.columns))
                
                if st.button("Process CSV Data"):
                    # Rename columns according to mapping
                    processed_data = data.copy()
                    for new_col, old_col in column_mapping.items():
                        if old_col != 'None' and old_col in data.columns:
                            processed_data[new_col] = data[old_col]
                    
                    # Add calculated power if not present
                    if 'panel_power' not in processed_data.columns:
                        processed_data['panel_power'] = (processed_data['panel_voltage'] * processed_data['panel_current'])
                    
                    # Add datetime if not present
                    if 'datetime' not in processed_data.columns or column_mapping['datetime'] == 'None':
                        processed_data['datetime'] = pd.date_range(start='2024-01-01', 
                                                                  periods=len(processed_data), 
                                                                  freq='H')
                    
                    data = processed_data
                    st.session_state['analysis_data'] = data
                    st.success("CSV data processed successfully!")
                    
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Perform analysis if data is available
    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
        
        if st.button("🔍 Analyze Performance", type="primary"):
            analyze_performance(data)

def analyze_performance(data):
    """Enhanced performance analysis function"""
    
    # Initialize anomaly detector
    detector = EnhancedAnomalyDetector(contamination=0.1)
    
    with st.spinner("Analyzing panel performance..."):
        # Detect anomalies
        analyzed_data = detector.detect_anomalies(data)
        
        # Analyze panel health
        panel_health = detector.analyze_panel_health(analyzed_data)
    
    # Display results
    st.success("Analysis completed!")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_panels = len(data['panel_id'].unique())
    total_anomalies = analyzed_data['is_anomaly'].sum()
    critical_panels = sum(1 for p in panel_health.values() if p['health_status'] == 'Critical')
    avg_output = data['energy_output'].mean()
    
    with col1:
        st.metric("Total Panels", total_panels)
    with col2:
        st.metric("Anomalies Detected", total_anomalies)
    with col3:
        st.metric("Critical Panels", critical_panels, delta=f"-{critical_panels}" if critical_panels > 0 else "0")
    with col4:
        st.metric("Avg Output (kWh)", f"{avg_output:.2f}")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Panel Health", "📈 Performance Trends", "💡 Recommendations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Anomaly Detection Results")
            fig = px.scatter(analyzed_data, x='datetime', y='energy_output',
                           color='is_anomaly', 
                           hover_data=['panel_id', 'panel_voltage', 'panel_current'],
                           title="Energy Output with Anomalies",
                           color_discrete_map={True: 'red', False: 'blue'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Panel Performance Distribution")
            panel_avg = analyzed_data.groupby('panel_id')['energy_output'].mean().reset_index()
            fig = px.box(analyzed_data, x='panel_id', y='energy_output',
                        title="Energy Output Distribution by Panel")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Panel Health Status")
        
        # Create health status dataframe
        health_df = pd.DataFrame.from_dict(panel_health, orient='index').reset_index()
        health_df = health_df.rename(columns={'index': 'panel_id'})
        health_df = health_df.sort_values('priority')
        
        # Color coding based on health status
        def color_health_status(val):
            if val == 'Critical':
                return 'background-color: #f8d7da'
            elif val == 'Poor':
                return 'background-color: #fff3cd'
            elif val == 'Fair':
                return 'background-color: #d1ecf1'
            else:
                return 'background-color: #d4edda'
        
        styled_df = health_df.style.applymap(color_health_status, subset=['health_status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Health status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            health_counts = health_df['health_status'].value_counts()
            fig = px.pie(values=health_counts.values, names=health_counts.index,
                        title="Panel Health Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(health_df, x='panel_id', y='anomaly_rate',
                        color='health_status', title="Anomaly Rate by Panel")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Trends")
        
        # Time series analysis
        if 'datetime' in analyzed_data.columns:
            daily_performance = analyzed_data.groupby([
                analyzed_data['datetime'].dt.date, 'panel_id'
            ])['energy_output'].sum().reset_index()
            daily_performance['datetime'] = pd.to_datetime(daily_performance['datetime'])
            
            fig = px.line(daily_performance, x='datetime', y='energy_output',
                         color='panel_id', title="Daily Energy Output Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance correlation matrix
            numeric_cols = ['energy_output', 'panel_voltage', 'panel_current', 'panel_power']
            available_cols = [col for col in numeric_cols if col in analyzed_data.columns]
            
            if len(available_cols) > 1:
                correlation_matrix = analyzed_data[available_cols].corr()
                fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                              title="Performance Metrics Correlation")
                st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency trends
        if 'irradiance' in analyzed_data.columns:
            analyzed_data['efficiency'] = analyzed_data['energy_output'] / (analyzed_data['irradiance'] * 0.01)
            efficiency_trend = analyzed_data.groupby('panel_id')['efficiency'].mean().reset_index()
            
            fig = px.bar(efficiency_trend, x='panel_id', y='efficiency',
                        title="Panel Efficiency (Energy/Irradiance)")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Maintenance Recommendations")
        
        # Priority-based recommendations
        critical_panels = [panel for panel, info in panel_health.items() 
                          if info['health_status'] == 'Critical']
        poor_panels = [panel for panel, info in panel_health.items() 
                      if info['health_status'] == 'Poor']
        fair_panels = [panel for panel, info in panel_health.items() 
                      if info['health_status'] == 'Fair']
        
        if critical_panels:
            st.markdown(f"""
            <div class="performance-danger">
                <h4>🚨 URGENT - Critical Issues Detected</h4>
                <p><strong>Panels requiring immediate attention:</strong> {', '.join(critical_panels)}</p>
                <p><strong>Recommended actions:</strong></p>
                <ul>
                    <li>Immediate inspection and maintenance</li>
                    <li>Check electrical connections and wiring</li>
                    <li>Inspect for physical damage or debris</li>
                    <li>Consider panel replacement if necessary</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if poor_panels:
            st.markdown(f"""
            <div class="performance-warning">
                <h4>⚠️ WARNING - Poor Performance Detected</h4>
                <p><strong>Panels needing maintenance:</strong> {', '.join(poor_panels)}</p>
                <p><strong>Recommended actions:</strong></p>
                <ul>
                    <li>Schedule maintenance within 1-2 weeks</li>
                    <li>Clean panel surfaces</li>
                    <li>Check inverter performance</li>
                    <li>Monitor closely for further degradation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if fair_panels:
            st.markdown(f"""
            <div class="performance-warning">
                <h4>ℹ️ CAUTION - Fair Performance</h4>
                <p><strong>Panels for monitoring:</strong> {', '.join(fair_panels)}</p>
                <p><strong>Recommended actions:</strong></p>
                <ul>
                    <li>Increase monitoring frequency</li>
                    <li>Schedule routine maintenance</li>
                    <li>Check for minor issues</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        good_panels = [panel for panel, info in panel_health.items() 
                      if info['health_status'] == 'Good']
        
        if good_panels:
            st.markdown(f"""
            <div class="performance-good">
                <h4>✅ GOOD - Normal Performance</h4>
                <p><strong>Well-performing panels:</strong> {', '.join(good_panels)}</p>
                <p><strong>Recommended actions:</strong></p>
                <ul>
                    <li>Continue regular monitoring</li>
                    <li>Maintain standard cleaning schedule</li>
                    <li>Annual performance review</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed recommendations based on data analysis
        st.subheader("Detailed Analysis & Recommendations")
        
        for panel_id, info in sorted(panel_health.items(), key=lambda x: x[1]['priority']):
            with st.expander(f"{panel_id} - {info['health_status']} Status"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Anomaly Rate", f"{info['anomaly_rate']:.1f}%")
                    st.metric("Average Output", f"{info['avg_output']:.2f} kWh")
                    st.metric("Total Readings", info['total_readings'])
                
                with col2:
                    st.metric("Anomaly Count", info['anomaly_count'])
                    st.metric("Output Stability", f"{info['output_stability']:.2f}")
                    st.metric("Voltage Stability", f"{info['voltage_stability']:.2f}")
                
                # Specific recommendations based on metrics
                recommendations = []
                
                if info['anomaly_rate'] > 20:
                    recommendations.append("⚠️ Very high anomaly rate - requires immediate inspection")
                elif info['anomaly_rate'] > 10:
                    recommendations.append("⚠️ High anomaly rate - schedule maintenance soon")
                
                if info['output_stability'] > info['avg_output'] * 0.3:
                    recommendations.append("📊 High output variability - check for intermittent issues")
                
                if info['voltage_stability'] > 2.0:
                    recommendations.append("⚡ Voltage instability detected - check electrical connections")
                
                if info['avg_output'] < 2.0:  # Assuming minimum expected output
                    recommendations.append("📉 Low average output - panel may be underperforming")
                
                if not recommendations:
                    recommendations.append("✅ Panel performing within normal parameters")
                
                for rec in recommendations:
                    st.write(f"• {rec}")

def simulator_page():
    st.header("Solar Configuration Simulator")
    st.write("Simulate different solar panel configurations to optimize performance.")

    # --- Step 1: Initialize session state to store results ---
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None

    # Configuration inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Panel Configuration")
        num_panels = st.slider("Number of Panels", 1, 50, 20, key="sim_panels")
        panel_wattage = st.slider("Panel Wattage (W)", 250, 500, 400, key="sim_wattage")
        tilt_angle = st.slider("Tilt Angle (degrees)", 0, 60, 30, key="sim_tilt")
    
    with col2:
        st.subheader("Location Settings")
        latitude = st.slider("Latitude", -90.0, 90.0, 21.1, key="sim_lat") # Default to Nagpur's latitude
        azimuth = st.slider("Azimuth (degrees)", 0, 360, 180, key="sim_azimuth")
        
        # Helper text for optimal Azimuth
        if latitude < 0:
            st.info("💡 For Southern Hemisphere locations, an Azimuth of **0° (North)** is optimal.")
        else:
            st.info("💡 For Northern Hemisphere locations, an Azimuth of **180° (South)** is optimal.")

        shading_factor = st.slider("Shading Factor (%)", 0, 50, 10, key="sim_shading")

    with col3:
        st.subheader("Maintenance Settings")
        cleaning_frequency = st.selectbox("Cleaning Frequency", 
                                        ["Weekly", "Monthly", "Quarterly", "Annually"], key="sim_cleaning")
        degradation_rate = st.slider("Annual Degradation (%)", 0.3, 1.0, 0.5, key="sim_degradation")

    if st.button("Run Simulation", type="primary"):
        # --- Step 2: Run simulation and STORE results in session state ---
        with st.spinner("Running simulations..."):
            configurations = []
            
            # Base configuration
            base_output = simulate_solar_output(num_panels, panel_wattage, tilt_angle, 
                                              latitude, azimuth, shading_factor, 
                                              cleaning_frequency, degradation_rate)
            configurations.append({
                'Configuration': 'Current',
                'Annual Output (kWh)': base_output,
                'Panels': num_panels,
                'Tilt': tilt_angle
            })
            
            # Optimized configurations (example variations)
            for tilt in [20, 35, 45]:
                if tilt != tilt_angle:
                    output = simulate_solar_output(num_panels, panel_wattage, tilt, latitude, azimuth, shading_factor, cleaning_frequency, degradation_rate)
                    configurations.append({'Configuration': f'Tilt {tilt}°', 'Annual Output (kWh)': output, 'Panels': num_panels, 'Tilt': tilt})
            
            for panels in [num_panels - 5, num_panels + 5]:
                if panels > 0:
                    output = simulate_solar_output(panels, panel_wattage, tilt_angle, latitude, azimuth, shading_factor, cleaning_frequency, degradation_rate)
                    configurations.append({'Configuration': f'{panels} Panels', 'Annual Output (kWh)': output, 'Panels': panels, 'Tilt': tilt_angle})
            
            df_configs = pd.DataFrame(configurations).round(2)
            st.session_state.simulation_results = df_configs

    # --- Step 3: ALWAYS try to display results if they exist in memory ---
    if st.session_state.simulation_results is not None:
        df_configs = st.session_state.simulation_results
        
        st.markdown("---")
        st.subheader("Simulation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Configuration Comparison")
            st.dataframe(df_configs)
            
            best_config = df_configs.loc[df_configs['Annual Output (kWh)'].idxmax()]
            st.success(f"🏆 Best Configuration: **{best_config['Configuration']}** ({best_config['Annual Output (kWh)']} kWh/year)")
        
        with col2:
            st.markdown("#### Performance Comparison")
            fig = px.bar(df_configs, x='Configuration', y='Annual Output (kWh)', title="Annual Energy Output by Configuration")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        # The ROI analysis is now part of the persistent display block
        st.subheader("Interactive ROI Analysis")
        electricity_rate = st.slider("Electricity Rate ($/kWh)", min_value=0.05, max_value=0.50, value=0.12, step=0.01)
        
        df_configs['Annual Savings ($)'] = df_configs['Annual Output (kWh)'] * electricity_rate
        df_configs['20-Year Savings ($)'] = df_configs['Annual Savings ($)'] * 20
        
        fig_roi = px.bar(df_configs, x='Configuration', y='20-Year Savings ($)',
                    title="Estimated 20-Year Financial Savings by Configuration",
                    labels={'20-Year Savings ($)': 'Savings ($)'})
        st.plotly_chart(fig_roi, use_container_width=True)

def simulate_solar_output(num_panels, panel_wattage, tilt_angle, latitude, 
                         azimuth, shading_factor, cleaning_frequency, degradation_rate):
    """Simulate solar output based on configuration parameters
    The Update: We'll replace the fixed 4.5 with a formula that dynamically calculates PSH based on the latitude input. 
    A good model is to assume PSH is highest at the equator and decreases as you move towards the poles.
    The Update: We will use the more physically accurate cosine function for both tilt and azimuth, and we'll make the azimuth calculation work globally.
    Tilt Efficiency: The loss will be calculated based on the cosine of the difference between the panel's tilt and the ideal tilt (which is often close to the latitude).
    Azimuth Efficiency: The code will first determine the optimal direction (180° South for the Northern Hemisphere, 0° North for the Southern) 
    and then use the cosine of the deviation from that optimal direction.
    """
    # Base calculation
    peak_sun_hours = 6.5 - 4 * (abs(latitude) / 90)
    base_output_per_panel = panel_wattage * peak_sun_hours 

    
    # Tilt angle optimization (simplified)
    tilt_difference = abs(tilt_angle - latitude)
    tilt_efficiency = math.cos(math.radians(tilt_difference))
    
    # Determines optimal direction (South=180° in North, North=0° in South).
    optimal_azimuth = 180 if latitude >= 0 else 0
    azimuth_difference = abs(azimuth - optimal_azimuth)
    # Cap difference at 90 degrees, as facing more than 90 degrees away is effectively the same as 90.
    if azimuth_difference > 90:
        azimuth_difference = 90
    azimuth_efficiency = math.cos(math.radians(azimuth_difference))
    
    # Shading losses
    shading_efficiency = 1 - (shading_factor / 100)
    
    # Cleaning frequency impact
    cleaning_efficiency = {
        'Weekly': 0.98,
        'Monthly': 0.95,
        'Quarterly': 0.90,
        'Annually': 0.85
    }[cleaning_frequency]
    
    # Annual degradation
    annual_efficiency = 1 - (degradation_rate / 100)
    
    # Calculate total annual output
    total_efficiency = (tilt_efficiency * azimuth_efficiency * shading_efficiency * cleaning_efficiency * annual_efficiency)
    
    annual_output = (base_output_per_panel * num_panels * 365 * total_efficiency) / 1000
    
    return annual_output

def data_upload_page():
    st.header("Data Upload & Analysis")
    
    st.write("Upload your solar panel data for custom analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Summary")
            st.write(df.describe())
            
            # Column selection for analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Select X-axis", numeric_columns)
                
                with col2:
                    y_axis = st.selectbox("Select Y-axis", numeric_columns)
                
                # Create visualization
                if st.button("Create Visualization"):
                    fig = px.scatter(df, x=x_axis, y=y_axis, 
                                   title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation analysis
                    correlation = df[numeric_columns].corr()
                    fig_corr = px.imshow(correlation, text_auto=True, 
                                       title="Correlation Matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        sample_data = {
            'datetime': ['2024-01-01 08:00:00', '2024-01-01 09:00:00', '2024-01-01 10:00:00'],
            'panel_id': ['Panel_01', 'Panel_01', 'Panel_01'],
            'irradiance': [600, 800, 1000],
            'temperature': [20, 22, 25],
            'energy_output': [2.4, 3.2, 4.0],
            'panel_voltage': [24.1, 24.3, 24.5],
            'panel_current': [2.1, 2.8, 3.5]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()