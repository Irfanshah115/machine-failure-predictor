import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="🔧",
    layout="centered"
)

# ============================================
# LOAD TRAINED MODEL
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load('best_failure_predictor.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_model()
    model_loaded = True
except:
    model_loaded = False

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/maintenance.png", width=80)
    st.title("About")
    st.write("""
    This AI-powered dashboard predicts whether industrial equipment 
    will fail within the next 30 days based on sensor readings.

    **Built by:** Irfan  
    **Course:** Zafar Iqbal ML Course  
    **Model:** Random Forest Classifier
    """)

    st.divider()
    st.subheader("How It Works")
    st.write("""
    1. Enter equipment sensor readings
    2. Select equipment type
    3. Click **Predict**
    4. AI tells you if maintenance is needed
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

if not model_loaded:
    st.error("⚠️ Model not found! Please train and save the model first.")
    st.info("Run the training notebook to generate: `best_failure_predictor.pkl`, `scaler.pkl`, `feature_columns.pkl`")
    st.stop()

st.success("✅ Model loaded successfully!")

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
        help="Select the type of equipment you want to analyze"
    )

    vibration = st.number_input(
        "Vibration (mm/s)",
        min_value=0.0, max_value=20.0, value=3.0, step=0.1,
        help="ISO 10816 standard. Normal: < 2.8 | Alert: 2.8-7.1 | Danger: > 7.1"
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=20.0, max_value=150.0, value=70.0, step=0.5,
        help="Bearing or casing temperature. Normal motors: < 80°C"
    )

    # Oil pressure only for equipment that has it
    if equipment_type in ['Centrifugal_Pump', 'Air_Compressor', 'Mixing_Tank_Agitator']:
        oil_pressure = st.number_input(
            "Oil Pressure (bar)",
            min_value=0.0, max_value=15.0, value=3.0, step=0.1,
            help="Lubrication system pressure. Typical: 2-6 bar"
        )
    else:
        oil_pressure = 0.0
        st.info("ℹ️ Oil pressure not applicable for this equipment type (set to 0)")

with col2:
    st.subheader("Maintenance History")

    running_hours = st.number_input(
        "Total Running Hours",
        min_value=0, max_value=50000, value=5000, step=100,
        help="Total operating hours since installation"
    )

    days_since_service = st.number_input(
        "Days Since Last Service",
        min_value=0, max_value=365, value=30, step=1,
        help="Days since last preventive maintenance (PM)"
    )

    # Visual gauge
    st.subheader("📈 Health Gauge")

    # Simple health score calculation
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

    # Prepare input data
    input_data = {
        'Vibration_mm_s': vibration,
        'Temperature_C': temperature,
        'Oil_Pressure_bar': oil_pressure,
        'Running_Hours': running_hours,
        'Days_Since_Last_Service': days_since_service
    }

    # Add equipment type dummy columns
    for col in feature_cols:
        if col.startswith('Equipment_Type_'):
            part_type = col.replace('Equipment_Type_', '')
            input_data[col] = 1 if equipment_type == part_type else 0

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])[feature_cols]

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

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

    # Probability bar chart
    st.subheader("Failure Probability Breakdown")
    prob_df = pd.DataFrame({
        'Status': ['Healthy (Next 30 Days)', 'Will Fail (Next 30 Days)'],
        'Probability': [prediction_proba[0], prediction_proba[1]]
    })
    st.bar_chart(prob_df.set_index('Status'), color=['#00CC66', '#FF4444'])

    # Show input summary
    st.subheader("📋 Input Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write(f"**Equipment:** {equipment_type}")
        st.write(f"**Vibration:** {vibration} mm/s")
        st.write(f"**Temperature:** {temperature} °C")
    with summary_col2:
        st.write(f"**Oil Pressure:** {oil_pressure} bar")
        st.write(f"**Running Hours:** {running_hours:,}")
        st.write(f"**Days Since Service:** {days_since_service}")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("Built with ❤️ by Irfan | Powered by Streamlit & Scikit-Learn | Zafar Iqbal ML Course 2026")
