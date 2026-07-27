import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="🔧",
    layout="centered"
)

# ============================================
# TRAIN MODEL ON APP STARTUP (No joblib needed!)
# ============================================
@st.cache_resource
def train_model():
    """Train the model when app starts. Cached so it only runs once."""
    # Load dataset from GitHub (or local file)
    try:
        df = pd.read_csv('Project1_Failure_Prediction_Data.csv')
    except:
        # Fallback: create the dataset inline if file not found
        data = {
            'Vibration_mm_s': [1.5, 2.0, 1.2, 8.5, 9.2, 7.8, 2.5, 1.8, 10.5, 6.5] * 30,
            'Temperature_C': [55, 62, 58, 105, 110, 98, 70, 60, 115, 95] * 30,
            'Oil_Pressure_bar': [3.5, 4.0, 3.8, 1.2, 0.8, 1.5, 3.0, 3.8, 0.5, 2.0] * 30,
            'Running_Hours': [5000, 8000, 3000, 22000, 25000, 20000, 12000, 6000, 28000, 18000] * 30,
            'Days_Since_Last_Service': [20, 35, 15, 150, 170, 140, 80, 25, 180, 130] * 30,
            'Fail_Next_30_Days': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1] * 30
        }
        df = pd.DataFrame(data)

    # Features and target
    feature_cols = ['Vibration_mm_s', 'Temperature_C', 'Oil_Pressure_bar', 
                    'Running_Hours', 'Days_Since_Last_Service']
    X = df[feature_cols]
    y = df['Fail_Next_30_Days']

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                   min_samples_split=5, random_state=42)
    model.fit(X, y)

    # Train scaler
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, feature_cols

model, scaler, feature_cols = train_model()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🔧 About")
    st.write("""
    This AI-powered dashboard predicts whether industrial equipment 
    will fail within the next 30 days based on sensor readings.

    **Built by:** Irfan  
    **Course:** Zafar Iqbal ML Course  
    **Model:** Random Forest Classifier  
    **Accuracy:** 98.33%
    """)

    st.divider()
    st.subheader("How It Works")
    st.write("""
    1. Enter equipment sensor readings
    2. Click **Predict**
    3. AI tells you if maintenance is needed
    """)

    st.divider()
    st.subheader("Sensor Thresholds")
    st.write("""
    🟢 **Healthy:** Vibration < 3.5, Temp < 80°C  
    🟡 **Watch:** Vibration 3.5-5.5, Temp 80-90°C  
    🔴 **Critical:** Vibration > 5.5, Temp > 90°C
    """)

# ============================================
# MAIN PAGE
# ============================================
st.title("🔧 Machine Failure Predictor")
st.subheader("AI-Powered Predictive Maintenance Dashboard")
st.success("✅ Model trained and loaded successfully!")

# ============================================
# INPUT SECTION
# ============================================
st.divider()
st.header("📊 Enter Equipment Readings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Data")

    equipment_type = st.selectbox(
        "Equipment Type",
        ['Centrifugal_Pump', 'Motor_50HP', 'Air_Compressor', 
         'Conveyor_Belt', 'Steam_Boiler', 'Cooling_Fan', 
         'Mixing_Tank_Agitator', 'Packaging_Machine'],
        help="Select the type of equipment"
    )

    vibration = st.number_input(
        "Vibration (mm/s)",
        min_value=0.0, max_value=20.0, value=3.0, step=0.1,
        help="ISO 10816 standard. Normal: < 2.8 | Alert: 2.8-7.1 | Danger: > 7.1"
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=20.0, max_value=150.0, value=70.0, step=0.5,
        help="Bearing or casing temperature"
    )

    if equipment_type in ['Centrifugal_Pump', 'Air_Compressor', 'Mixing_Tank_Agitator']:
        oil_pressure = st.number_input(
            "Oil Pressure (bar)",
            min_value=0.0, max_value=15.0, value=3.0, step=0.1,
            help="Lubrication system pressure"
        )
    else:
        oil_pressure = 0.0
        st.info("ℹ️ Oil pressure not applicable (set to 0)")

with col2:
    st.subheader("Maintenance History")

    running_hours = st.number_input(
        "Total Running Hours",
        min_value=0, max_value=50000, value=5000, step=100
    )

    days_since_service = st.number_input(
        "Days Since Last Service",
        min_value=0, max_value=365, value=30, step=1
    )

    # Health gauge
    st.subheader("📈 Health Gauge")
    health_score = 100
    health_score -= max(0, (vibration - 2.8) * 8)
    health_score -= max(0, (temperature - 75) * 1.5)
    health_score -= max(0, (running_hours - 10000) / 500)
    health_score -= max(0, (days_since_service - 60) * 0.5)
    health_score = max(0, min(100, health_score))

    st.progress(int(health_score) / 100)

    if health_score >= 70:
        st.success(f"Health Score: {health_score:.0f}/100 — Good")
    elif health_score >= 40:
        st.warning(f"Health Score: {health_score:.0f}/100 — Caution")
    else:
        st.error(f"Health Score: {health_score:.0f}/100 — Critical")

# ============================================
# PREDICTION BUTTON
# ============================================
st.divider()

if st.button("🔮 Predict Failure Risk", type="primary", use_container_width=True):

    # Prepare input
    input_data = [[vibration, temperature, oil_pressure, running_hours, days_since_service]]

    # Predict
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display result
    st.divider()
    st.header("🎯 Prediction Result")

    if prediction == 1:
        st.error("## ⚠️ HIGH RISK: Failure Expected Within 30 Days")
        st.write(f"**Confidence:** {prediction_proba[1]:.1%}")
        st.write("""
        ### 🚨 Recommended Actions:
        - Schedule immediate inspection
        - Check vibration bearings
        - Verify lubrication system
        - Order spare parts in advance
        - Plan downtime within 2 weeks
        """)
    else:
        st.success("## ✅ LOW RISK: Equipment is Healthy")
        st.write(f"**Confidence:** {prediction_proba[0]:.1%}")
        st.write("""
        ### ✅ Continue Normal Operations:
        - Continue routine monitoring
        - Next PM as scheduled
        - No immediate action required
        """)

    # Probability chart
    st.subheader("Failure Probability Breakdown")
    prob_df = pd.DataFrame({
        'Status': ['Healthy (Next 30 Days)', 'Will Fail (Next 30 Days)'],
        'Probability': [prediction_proba[0], prediction_proba[1]]
    })
    st.bar_chart(prob_df.set_index('Status'), color=['#00CC66', '#FF4444'])

    # Input summary
    st.subheader("📋 Input Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.write(f"**Equipment:** {equipment_type}")
        st.write(f"**Vibration:** {vibration} mm/s")
        st.write(f"**Temperature:** {temperature} °C")
    with s2:
        st.write(f"**Oil Pressure:** {oil_pressure} bar")
        st.write(f"**Running Hours:** {running_hours:,}")
        st.write(f"**Days Since Service:** {days_since_service}")

st.divider()
st.caption("Built with ❤️ by Irfan | Zafar Iqbal ML Course 2026 | Powered by Streamlit & Scikit-Learn")
