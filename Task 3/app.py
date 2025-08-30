import streamlit as st
import pickle
import numpy as np

# --- Load models ---
rf_model= pickle.load(open("rf_model.pkl", "rb"))
xgb_model=pickle.load(open("xgb_model.pkl", "rb"))

# --- Page title ---
st.set_page_config(page_title="Forest Cover Type Prediction", layout="wide")
st.title("🌲 Predict Forest Cover Type")

st.markdown("Enter the environmental values so the model can predict the forest cover type 🌳")

# --- Input values (Sliders) ---
col1, col2 = st.columns(2)

with col1:
    elevation = st.slider("Elevation (meters)", 1500, 4000, 2500)
    aspect = st.slider("Aspect (degrees)", 0, 360, 180)
    slope = st.slider("Slope (degrees)", 0, 60, 15)
    hillshade = st.slider("Hillshade 9am", 0, 300, 200)

with col2:
    distance_roadways = st.slider("Distance to Roadways (m)", 0, 7000, 2000)
    distance_fire = st.slider("Distance to Fire Points (m)", 0, 7000, 3000)

# --- Prepare data ---
features = np.array([[elevation, aspect, slope, hillshade, distance_roadways, distance_fire] + [0]*14])

# --- Prediction ---
if st.button("Predict"):
    rf_pred = rf_model.predict(features)[0]
    rf_proba = max(rf_model.predict_proba(features)[0]) * 100

    xgb_pred = xgb_model.predict(features)[0]
    xgb_proba = max(xgb_model.predict_proba(features)[0]) * 100

    # --- Display results ---
    st.subheader("🎯 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest Prediction", f"Type {rf_pred}", f"{rf_proba:.1f}% confidence")
    with col2:
        st.metric("XGBoost Prediction", f"Type {xgb_pred}", f"{xgb_proba:.1f}% confidence")