import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Machine Failure Predictor", page_icon="🔧", layout="centered")

# ============================================
# TRAIN MODEL ON STARTUP
# ============================================
@st.cache_resource
def get_model():
    try:
        df = pd.read_csv('Project1_Failure_Prediction_Data.csv')
    except:
        # Inline dataset if file not found
        np.random.seed(42)
        n = 300
        data = {
            'Vibration_mm_s': list(np.random.uniform(0.5, 3.5, 150)) + list(np.random.uniform(5.5, 12.0, 150)),
            'Temperature_C': list(np.random.uniform(45, 78, 150)) + list(np.random.uniform(90, 130, 150)),
            'Oil_Pressure_bar': list(np.random.uniform(2.5, 5.0, 150)) + list(np.random.uniform(0.2, 1.8, 150)),
            'Running_Hours': list(np.random.randint(500, 12000, 150)) + list(np.random.randint(17000, 30000, 150)),
            'Days_Since_Last_Service': list(np.random.randint(5, 70, 150)) + list(np.random.randint(125, 200, 150)),
            'Fail_Next_30_Days': [0]*150 + [1]*150
        }
        df = pd.DataFrame(data)

    X = df[['Vibration_mm_s', 'Temperature_C', 'Oil_Pressure_bar', 'Running_Hours', 'Days_Since_Last_Service']]
    y = df['Fail_Next_30_Days']
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X, y)
    return model

model = get_model()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🔧 About")
    st.write("AI-powered predictive maintenance dashboard.")
    st.write("**Built by:** Irfan")
    st.write("**Course:** Zafar Iqbal ML")
    st.write("**Model:** Random Forest")
    st.divider()
    st.subheader("Thresholds")
    st.write("🟢 Healthy: Vib < 3.5, Temp < 80°C")
    st.write("🟡 Watch: Vib 3.5-5.5, Temp 80-90°C")
    st.write("🔴 Critical: Vib > 5.5, Temp > 90°C")

# ============================================
# MAIN PAGE
# ============================================
st.title("🔧 Machine Failure Predictor")
st.subheader("Predictive Maintenance Dashboard")
st.success("✅ Model loaded!")

st.divider()
st.header("📊 Enter Equipment Readings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Data")
    equipment_type = st.selectbox("Equipment Type", 
        ['Centrifugal_Pump', 'Motor_50HP', 'Air_Compressor', 'Conveyor_Belt', 
         'Steam_Boiler', 'Cooling_Fan', 'Mixing_Tank_Agitator', 'Packaging_Machine'])

    vibration = st.number_input("Vibration (mm/s)", 0.0, 20.0, 3.0, 0.1)
    temperature = st.number_input("Temperature (°C)", 20.0, 150.0, 70.0, 0.5)

    if equipment_type in ['Centrifugal_Pump', 'Air_Compressor', 'Mixing_Tank_Agitator']:
        oil_pressure = st.number_input("Oil Pressure (bar)", 0.0, 15.0, 3.0, 0.1)
    else:
        oil_pressure = 0.0
        st.info("Oil pressure N/A")

with col2:
    st.subheader("History")
    running_hours = st.number_input("Running Hours", 0, 50000, 5000, 100)
    days_since_service = st.number_input("Days Since Service", 0, 365, 30, 1)

    st.subheader("Health Score")
    score = 100
    score -= max(0, (vibration - 2.8) * 8)
    score -= max(0, (temperature - 75) * 1.5)
    score -= max(0, (running_hours - 10000) / 500)
    score -= max(0, (days_since_service - 60) * 0.5)
    score = max(0, min(100, score))
    st.progress(int(score))
    if score >= 70: st.success(f"{score:.0f}/100 — Good")
    elif score >= 40: st.warning(f"{score:.0f}/100 — Caution")
    else: st.error(f"{score:.0f}/100 — Critical")

# ============================================
# PREDICT
# ============================================
st.divider()
if st.button("🔮 Predict Failure Risk", type="primary", use_container_width=True):
    input_data = [[vibration, temperature, oil_pressure, running_hours, days_since_service]]
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.divider()
    st.header("🎯 Result")

    if pred == 1:
        st.error("## ⚠️ HIGH RISK: Failure Expected Within 30 Days")
        st.write(f"**Confidence:** {proba[1]:.1%}")
        st.write("🚨 Schedule inspection immediately!")
        st.write("🚨 Check bearings and lubrication!")
        st.write("🚨 Order spare parts in advance!")
    else:
        st.success("## ✅ LOW RISK: Equipment is Healthy")
        st.write(f"**Confidence:** {proba[0]:.1%}")
        st.write("✅ Continue normal operations")
        st.write("✅ Next PM as scheduled")

    st.subheader("Probability Breakdown")
    chart_data = pd.DataFrame({
        'Status': ['Healthy', 'Will Fail'],
        'Probability': [proba[0], proba[1]]
    })
    st.bar_chart(chart_data.set_index('Status'))

    st.subheader("Input Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Equipment: {equipment_type}")
        st.write(f"Vibration: {vibration} mm/s")
        st.write(f"Temperature: {temperature} °C")
    with c2:
        st.write(f"Oil Pressure: {oil_pressure} bar")
        st.write(f"Running Hours: {running_hours:,}")
        st.write(f"Days Since Service: {days_since_service}")

st.divider()
st.caption("Built by Irfan | Zafar Iqbal ML Course 2026")
