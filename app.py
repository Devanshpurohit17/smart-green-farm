# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------
# Config / paths
# -----------------------------
BASE_DIR = Path(__file__).parent
MODEL_FILENAME = "hybrid_crop_reco_model.pkl"
DATA_FILENAME = "Crop_recommendation.csv"

# -----------------------------
# Page config + CSS
# -----------------------------
st.set_page_config(page_title="Smart Green Farm - Crop Recommendation", layout="wide", page_icon="🌿")

CUSTOM_CSS = """
<style>
.stApp { background: linear-gradient(135deg, #eaf7ec 0%, #f7fff8 100%); }
.card { background-color: rgba(255,255,255,0.95); padding: 18px; border-radius: 12px; }
.result { background: linear-gradient(135deg,#c8f2d0,#78d28f); padding: 14px; border-radius: 12px; font-weight:600; }
.small { font-size:13px; color:#2b6b2b; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helper: load model + dataset stats (cached)
# -----------------------------
@st.cache_resource
def load_model_and_stats(model_path: Path, csv_path: Path):
    # Load model
    model = joblib.load(model_path)
    # Load CSV used for training (to get ranges)
    df = pd.read_csv(csv_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    # Basic check
    if not set(feature_cols).issubset(df.columns):
        raise ValueError(f"CSV does not contain required feature columns: {feature_cols}. Found: {df.columns.tolist()}")
    stats = df[feature_cols].describe()
    return model, stats

# Try load model + stats, show friendly error if not present
try:
    model_path = BASE_DIR / MODEL_FILENAME
    csv_path = BASE_DIR / DATA_FILENAME
    model, stats = load_model_and_stats(model_path, csv_path)
except Exception as e:
    st.error("Model or dataset could not be loaded. Check files and paths.")
    st.exception(e)
    st.stop()

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def rng(col):
    return float(stats.loc["min", col]), float(stats.loc["max", col]), float(stats.loc["mean", col])

# -----------------------------
# Database helpers
# -----------------------------
DB_URL = None
try:
    DB_URL = st.secrets["DATABASE_URL"]
except Exception:
    DB_URL = None

ENGINE = None
DB_CONNECTED = False
DB_ERROR_TEXT = None

if DB_URL:
    try:
        # enforce sslmode require in engine connect args (helps with Supabase)
        ENGINE = create_engine(DB_URL, pool_pre_ping=True, connect_args={"sslmode": "require"})
        # Test a quick connection & create table if not exists
        with ENGINE.begin() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.crop_submissions (
                id SERIAL PRIMARY KEY,
                submitted_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                n INTEGER,
                p INTEGER,
                k INTEGER,
                temperature REAL,
                humidity REAL,
                ph REAL,
                rainfall REAL,
                predicted_crop TEXT
            );
            """))
        DB_CONNECTED = True
    except SQLAlchemyError as e:
        DB_ERROR_TEXT = str(e.__cause__ or e)
        DB_CONNECTED = False
    except Exception as e:
        DB_ERROR_TEXT = str(e)
        DB_CONNECTED = False
else:
    DB_CONNECTED = False
    DB_ERROR_TEXT = "DATABASE_URL not set in Streamlit secrets."

# -----------------------------
# Sidebar status + tips
# -----------------------------
with st.sidebar:
    st.title("Smart Green Farm")
    st.markdown("AI crop recommendation (model A).")
    st.markdown("---")
    st.markdown("**Feature Ranges (from training data)**")
    for col in feature_cols:
        mn, mx, mean = rng(col)
        st.write(f"- **{col}** → {mn:.2f} to {mx:.2f} (mean {mean:.2f})")
    st.markdown("---")

    if DB_CONNECTED:
        st.success("Connected to cloud DB (submissions will be saved).")
    else:
        st.warning("No DB connection.")
        st.write("Tip: set `DATABASE_URL` in Streamlit secrets (TOML) with `?sslmode=require` added.")
        if DB_ERROR_TEXT:
            st.caption("DB error:")
            st.code(DB_ERROR_TEXT, language="")

    st.markdown("---")
    st.markdown("Made for AIML project. Use values within dataset ranges for reliable suggestions.")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<h1>Smart Green Farm – Crop Recommendation</h1>", unsafe_allow_html=True)
st.mark.markdown = st.markdown  # avoid lint noise
left, right = st.columns([2, 1])

with left:
    st.subheader("Field and Soil Conditions")
    N_min, N_max, N_mean = rng("N")
    P_min, P_max, P_mean = rng("P")
    K_min, K_max, K_mean = rng("K")
    T_min, T_max, T_mean = rng("temperature")
    H_min, H_max, H_mean = rng("humidity")
    pH_min, pH_max, pH_mean = rng("ph")
    R_min, R_max, R_mean = rng("rainfall")

    N = st.slider("Nitrogen (N)", int(N_min), int(N_max), int(N_mean))
    P = st.slider("Phosphorus (P)", int(P_min), int(P_max), int(P_mean))
    K = st.slider("Potassium (K)", int(K_min), int(K_max), int(K_mean))
    temperature = st.slider("Temperature (°C)", float(T_min), float(T_max), float(T_mean), step=0.5)
    humidity = st.slider("Humidity (%)", float(H_min), float(H_max), float(H_mean), step=0.5)
    ph = st.slider("Soil pH", float(pH_min), float(pH_max), float(pH_mean), step=0.1)
    rainfall = st.slider("Rainfall (mm)", float(R_min), float(R_max), float(R_mean), step=1.0)

    predict_btn = st.button("Recommend Crop")

with right:
    st.subheader("Recommendation")
    result_area = st.empty()
    if predict_btn:
        try:
            inp_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_cols)
            crop = model.predict(inp_df)[0]
            result_area.markdown(f'<div class="result">Recommended crop: {crop}</div>', unsafe_allow_html=True)
            # Save to DB if connected
            if DB_CONNECTED and ENGINE is not None:
                try:
                    with ENGINE.begin() as conn:
                        conn.execute(
                            text("""
                                INSERT INTO public.crop_submissions
                                (n, p, k, temperature, humidity, ph, rainfall, predicted_crop)
                                VALUES (:n, :p, :k, :temperature, :humidity, :ph, :rainfall, :pred)
                            """),
                            {
                                "n": int(N), "p": int(P), "k": int(K),
                                "temperature": float(temperature),
                                "humidity": float(humidity),
                                "ph": float(ph),
                                "rainfall": float(rainfall),
                                "pred": str(crop)
                            }
                        )
                    st.success("Submission saved to database.")
                except Exception as e:
                    st.error("Could not save submission to DB.")
                    st.exception(e)
            else:
                st.info("DB not connected, submission not saved.")
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
    else:
        st.info("Adjust sliders and click **Recommend Crop**.")

# -----------------------------
# Recent submissions preview
# -----------------------------
st.markdown("---")
st.subheader("Recent Submissions (DB preview)")
if DB_CONNECTED and ENGINE is not None:
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text("SELECT id, submitted_at, n, p, k, temperature, humidity, ph, rainfall, predicted_crop FROM public.crop_submissions ORDER BY id DESC LIMIT 10")).mappings().all()
        if rows:
            df_preview = pd.DataFrame(rows)
            st.dataframe(df_preview, use_container_width=True)
        else:
            st.write("No submissions yet.")
    except Exception as e:
        st.error("Could not read submissions from DB.")
        st.exception(e)
else:
    st.info("Database not connected. To enable saving, add `DATABASE_URL` to Streamlit secrets and redeploy.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<small class='small'>Smart Green Farm prototype • Model A (Crop Recommendation) • Add Model B/C later</small>", unsafe_allow_html=True)
