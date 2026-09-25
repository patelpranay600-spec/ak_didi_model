import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import datetime

st.set_page_config(page_title="Smart Meter Anomaly Detector", layout="centered")

# Load model and artifacts once per session to optimize memory and speed
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('smart_meter_stacking_pipeline.joblib')
    except Exception as e:
        st.error(f"Failed to load pipeline artifacts: {e}")
        st.stop()

artifacts = load_pipeline()
model = artifacts['model']
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']
ANOMALY_THRESHOLD = artifacts['anomaly_threshold']

def preprocess_input(date_val, time_val, electricity_consumed, avg_past_consumption, temperature, humidity, wind_speed):
    # Combine date and time
    ts = pd.to_datetime(f"{date_val} {time_val}")
    
    # Time feature extraction
    time_of_day = ts.hour + (ts.minute / 60.0)
    day_of_week = ts.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    # Sine / Cosine Cyclical Encoding
    time_sin = math.sin(2 * math.pi * time_of_day / 24.0)
    time_cos = math.cos(2 * math.pi * time_of_day / 24.0)
    day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
    day_cos = math.cos(2 * math.pi * day_of_week / 7.0)

    # Calculate engineered features exactly as done in training
    consumption_deviation = electricity_consumed - avg_past_consumption
    consumption_ratio = electricity_consumed / (avg_past_consumption + 0.0001)

    # Construct dictionary matching exact feature names
    feature_dict = {
        'Electricity_Consumed': electricity_consumed,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Avg_Past_Consumption': avg_past_consumption,
        'Is_Weekend': is_weekend,
        'Time_Sin': time_sin,
        'Time_Cos': time_cos,
        'Day_Sin': day_sin,
        'Day_Cos': day_cos,
        'Consumption_Deviation': consumption_deviation,
        'Consumption_Ratio': consumption_ratio
    }

    # Convert to DataFrame and align columns exactly as trained
    df_features = pd.DataFrame([feature_dict])
    df_aligned = df_features[expected_features]

    # Standard scale
    return scaler.transform(df_aligned)
st.title("⚡ Smart Meter Anomaly Detector")
st.write("Run local predictions natively using the trained stacking classifier.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Time & Usage")
    date_input = st.date_input("Date", datetime.date(2024, 1, 1))
    time_input = st.time_input("Time", datetime.time(14, 30))
    electricity_consumed = st.slider("Electricity Consumed (Scaled)", 0.0, 1.0, 0.85)
    avg_past_consumption = st.slider("Avg Past Consumption (Scaled)", 0.0, 1.0, 0.52)

with col2:
    st.subheader("Weather Conditions")
    temperature = st.slider("Temperature (Scaled)", 0.0, 1.0, 0.45)
    humidity = st.slider("Humidity (Scaled)", 0.0, 1.0, 0.62)
    wind_speed = st.slider("Wind Speed (Scaled)", 0.0, 1.0, 0.31)

st.markdown("---")

if st.button("Analyze Reading", type="primary"):
    with st.spinner("Analyzing patterns..."):
        # Process the raw inputs through the exact training pipeline
        # Process the raw inputs through the exact training pipeline
        scaled_input = preprocess_input(
            date_input, time_input, electricity_consumed, 
            avg_past_consumption, temperature, humidity, wind_speed
        )
        
        # Predict probability
        probabilities = model.predict_proba(scaled_input)[0]
        anomaly_probability = float(probabilities[0])  # Restore Class 0 as Anomaly
        normal_probability = float(probabilities[1])
        # Evaluate against custom threshold
        is_anomaly = anomaly_probability >= ANOMALY_THRESHOLD
        
        if is_anomaly:
            st.error(f"🚨 **ANOMALY DETECTED** (Probability: {anomaly_probability:.2%})")
            st.write(f"This reading crosses the tuned threshold of {ANOMALY_THRESHOLD:.2%}.")
        else:
            st.success(f"✅ **NORMAL USAGE** (Probability: {normal_probability:.2%})")
            st.write("Consumption is well within expected parameters.")