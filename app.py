import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import os

# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "crop_model.pkl"   # <-- your saved model filename
model = joblib.load(MODEL_PATH)

# ---------------------------
# SUPABASE CONNECTION
# ---------------------------
db_url = os.getenv("DATABASE_URL")  # streamlit secrets inject this

engine = None
if db_url:
    try:
        engine = create_engine(
            db_url, 
            connect_args={"sslmode": "require"}, 
            pool_pre_ping=True
        )
    except Exception as e:
        st.sidebar.error("❌ DB connection failed.")
        st.sidebar.write(str(e))
else:
    st.sidebar.warning("⚠️ No DB connection. DATABASE_URL missing in secrets.")

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🌱 Smart Green Farm – Crop Recommendation")

st.markdown("Adjust soil conditions and click **Recommend Crop**.")

# ---------------------------
# USER INPUTS
# ---------------------------
N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 0, 140, 40)
K = st.slider("Potassium (K)", 0, 200, 50)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# Prepare input
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("🌾 Recommend Crop"):

    prediction = model.predict(input_data)[0]
    st.success(f"### ✅ Recommended Crop: **{prediction}**")

    # ---------------------------
    # SAVE to DATABASE
    # ---------------------------
    if engine:
        try:
            df = pd.DataFrame([{
                "nitrogen": N,
                "phosphorus": P,
                "potassium": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "prediction": prediction
            }])

            df.to_sql("crop_predictions", engine, if_exists="append", index=False)
            st.info("📡 Saved to Supabase successfully!")

        except Exception as e:
            st.error("❌ Failed to save data to Supabase.")
            st.write(str(e))

