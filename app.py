# app.py - Crop Recommendation (Model A) with optional Postgres saving (Supabase)
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st

# SQLAlchemy for DB
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Text, DateTime as SA_DateTime
)
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------
# CONFIG - paths (cloud-safe)
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_FILENAME = "Crop_recommendation.csv"         # update if your CSV name differs
MODEL_FILENAME = "hybrid_crop_reco_model.pkl"     # your trained model file

DATA_PATH = BASE_DIR / DATA_FILENAME
MODEL_PATH = BASE_DIR / MODEL_FILENAME

st.set_page_config(page_title="Smart Green Farm - Crop Recommendation",
                   page_icon="🌿", layout="wide")

# -----------------------------
# CSS (light accessible design)
# -----------------------------
CUSTOM_CSS = """
<style>
.stApp { background: linear-gradient(135deg,#e9f7ee 0%, #f3fbf9 100%); color: #0b3b18; font-family: "Segoe UI", sans-serif;}
.hero-title { font-size:28px; font-weight:700; color:#0b3b18; margin:18px 0; }
.card { background: rgba(255,255,255,0.95); border-radius:12px; padding:16px; box-shadow: 0 8px 20px rgba(0,0,0,0.06); }
.result-card { background: linear-gradient(135deg,#81c784,#43a047); color:#062e12; padding:14px; border-radius:12px; font-weight:700; text-align:center; }
.sidebar-title { font-weight:700; color:#0b3b18; }
.small-muted { color:#2f6b3a; font-size:13px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Load model + dataset safely
# -----------------------------
@st.cache_resource
def load_model_and_stats():
    # Check files
    if not DATA_PATH.exists():
        st.error(f"Missing dataset file: {DATA_FILENAME}. Put it in the same folder as app.py")
        st.stop()
    if not MODEL_PATH.exists():
        st.error(f"Missing model file: {MODEL_FILENAME}. Put it in the same folder as app.py")
        st.stop()

    # load CSV (same one used for training)
    data = pd.read_csv(DATA_PATH)

    # ensure expected feature columns exist (common crop features)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for c in feature_cols:
        if c not in data.columns:
            st.error(f"Required feature column '{c}' not found in CSV.")
            st.stop()

    stats = data[feature_cols].describe()
    model = joblib.load(MODEL_PATH)
    return model, stats, feature_cols, data

model, stats, feature_cols, full_data = load_model_and_stats()

def rng(col):
    col_min = float(stats.loc["min", col])
    col_max = float(stats.loc["max", col])
    col_mean = float(stats.loc["mean", col])
    return col_min, col_max, col_mean

# -----------------------------
# Database (Postgres) setup
# -----------------------------
ENGINE = None
metadata = MetaData()
submissions_table = None

# Get DB URL from Streamlit secrets or environment
db_url = None
try:
    if "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    elif "DATABASE" in st.secrets and "url" in st.secrets["DATABASE"]:
        db_url = st.secrets["DATABASE"]["url"]
except Exception:
    db_url = None

if not db_url:
    import os
    db_url = os.environ.get("DATABASE_URL")

if db_url:
    try:
        ENGINE = create_engine(db_url, pool_pre_ping=True)
        # Build submissions table schema dynamically from feature_cols
        cols = [
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("submitted_at", SA_DateTime, nullable=False),
            Column("source", String(64), nullable=True),
        ]
        # feature columns as floats
        for c in feature_cols:
            safe_name = c.replace(" ", "_").replace("-", "_")[:60]
            cols.append(Column(safe_name, Float))
        cols += [
            Column("predicted_crop", String(256)),
            Column("predicted_proba", Text),
            Column("notes", Text, nullable=True),
        ]
        submissions_table = Table("crop_submissions", metadata, *cols)
        metadata.create_all(ENGINE)
        connected_msg = "Connected to cloud DB (submissions will be saved)."
    except SQLAlchemyError as ex:
        ENGINE = None
        submissions_table = None
        connected_msg = f"DB connection failed: {str(ex)}"
else:
    connected_msg = "No DATABASE_URL found. Submissions will NOT be saved."

# helper to save
def save_submission(conn, inputs: dict, predicted_crop=None, proba=None, source="streamlit_app", notes=None):
    ins = {
        "submitted_at": datetime.utcnow(),
        "source": source,
        "predicted_crop": str(predicted_crop) if predicted_crop is not None else None,
        "predicted_proba": json.dumps(proba.tolist() if hasattr(proba, "tolist") else proba) if proba is not None else None,
        "notes": notes
    }
    for k, v in inputs.items():
        key = k.replace(" ", "_").replace("-", "_")[:60]
        try:
            ins[key] = float(v) if v is not None else None
        except Exception:
            ins[key] = None
    res = conn.execute(submissions_table.insert().values(**ins))
    try:
        return res.inserted_primary_key[0]
    except Exception:
        return None

# -----------------------------
# Sidebar + info
# -----------------------------
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'>🌿 Smart Green Farm</div>", unsafe_allow_html=True)
    st.write("Crop Recommendation (Model A) — enter field values and get a recommended crop.")
    st.markdown("---")
    st.write("<div class='small-muted'>Dataset feature ranges (training data):</div>", unsafe_allow_html=True)
    for col in feature_cols:
        mn, mx, _ = rng(col)
        st.write(f"- **{col}**: `{mn:.1f}` to `{mx:.1f}`")
    st.markdown("---")
    if ENGINE:
        st.success(connected_msg)
    else:
        st.info(connected_msg)

# -----------------------------
# Main UI
# -----------------------------
st.markdown('<div class="hero-title">Smart Green Farm – Crop Recommendation</div>', unsafe_allow_html=True)
st.write("Simulate your farm conditions (within training ranges) and click Recommend Crop.")

left_col, right_col = st.columns([2, 1])

# LEFT: inputs
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Field and Soil Conditions")

    # read ranges
    N_min, N_max, N_mean = rng("N")
    P_min, P_max, P_mean = rng("P")
    K_min, K_max, K_mean = rng("K")
    T_min, T_max, T_mean = rng("temperature")
    H_min, H_max, H_mean = rng("humidity")
    pH_min, pH_max, pH_mean = rng("ph")
    R_min, R_max, R_mean = rng("rainfall")

    # defensive checks and sliders
    def safe_slider(label, mn, mx, default, step_hint=None):
        # handle constant columns
        if mx <= mn:
            st.info(f"'{label}' appears constant in training data (value = {mn}).")
            val = st.number_input(label, value=float(mn), disabled=True)
            return float(val)
        # step selection
        rng_val = mx - mn
        if step_hint:
            step = step_hint
        else:
            if rng_val <= 10 and float(mn).is_integer() and float(mx).is_integer():
                step = 1.0
            else:
                step = max(rng_val / 40.0, 0.1)
        # clamp default
        default = float(max(min(default, mx), mn))
        return st.slider(label, min_value=float(mn), max_value=float(mx), value=default, step=float(step))

    N = safe_slider("Nitrogen (N)", N_min, N_max, N_mean, step_hint=1.0)
    P = safe_slider("Phosphorus (P)", P_min, P_max, P_mean, step_hint=1.0)
    K = safe_slider("Potassium (K)", K_min, K_max, K_mean, step_hint=1.0)
    temperature = safe_slider("Temperature (°C)", T_min, T_max, T_mean, step_hint=0.5)
    humidity = safe_slider("Humidity (%)", H_min, H_max, H_mean, step_hint=1.0)
    ph = safe_slider("Soil pH", pH_min, pH_max, pH_mean, step_hint=0.1)
    rainfall = safe_slider("Rainfall (mm)", R_min, R_max, R_mean, step_hint=1.0)

    st.markdown('</div>', unsafe_allow_html=True)
    predict_button = st.button("Recommend Crop", use_container_width=True)

# RIGHT: output
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recommendation")

    if predict_button:
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        df_in = pd.DataFrame(input_data, columns=feature_cols)
        try:
            crop = model.predict(df_in)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            crop = None

        if crop is not None:
            st.markdown(f'<div class="result-card">Recommended crop: {crop}</div>', unsafe_allow_html=True)

            # interpretation text
            st.markdown("#### Interpretation")
            st.write("Recommendation is based on training data ranges and the learned model patterns. Use local knowledge before applying in real field.")

            # probabilities if available
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df_in)[0]
                    prob_df = pd.DataFrame({"Crop": model.classes_, "Probability": np.round(proba, 3)})
                    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
                    st.markdown("#### Prediction Confidence")
                    st.table(prob_df)
                except Exception:
                    proba = None
            else:
                proba = None

            # Save submission to DB if configured
            if ENGINE and submissions_table is not None:
                try:
                    with ENGINE.begin() as conn:
                        new_id = save_submission(conn,
                                                 inputs=dict(zip(feature_cols, input_data[0])),
                                                 predicted_crop=crop,
                                                 proba=proba,
                                                 source="crop_reco_app")
                    if new_id:
                        st.success(f"Submission saved (id={new_id})")
                except Exception as e:
                    st.warning(f"Could not save submission to DB: {e}")
        else:
            st.info("No recommendation available.")
    else:
        st.write("Adjust the sliders and click **Recommend Crop** to get a suggestion.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Smart Green Farm prototype • Model A (Crop Recommendation)")
