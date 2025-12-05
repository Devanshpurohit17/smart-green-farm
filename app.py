# app.py
import json
from pathlib import Path
from datetime import datetime
import os

import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Text, DateTime as SA_DateTime
)
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------
# SAFE RELATIVE LOCATIONS
# -----------------------------
DATA_FILENAME = "Crop_recommendation.csv"
MODEL_FILENAME = "hybrid_crop_reco_model.pkl"

DATA_PATH = Path(DATA_FILENAME)
MODEL_PATH = Path(MODEL_FILENAME)

st.set_page_config(page_title="Smart Green Farm - Crop Recommendation",
                   page_icon="🌿", layout="wide")

# -----------------------------
# MODEL + DATA LOAD
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_FILENAME}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset missing: {DATA_FILENAME}")
        st.stop()
    data = pd.read_csv(DATA_PATH)
    feature_cols = ['N','P','K','temperature','humidity','ph','rainfall']
    for col in feature_cols:
        if col not in data.columns:
            st.error(f"Feature '{col}' missing in CSV")
            st.stop()
    stats = data[feature_cols].describe()

    if "min" not in stats.index or "max" not in stats.index:
        st.error("CSV is malformed — stats not available")
        st.stop()

    return data, stats, feature_cols

model = load_model()
full_data, stats, feature_cols = load_data()

def rng(col):
    mn = float(stats.loc["min", col])
    mx = float(stats.loc["max", col])
    mean = float(stats.loc["mean", col])
    return mn, mx, mean

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
ENGINE = None
metadata = MetaData()
submissions_table = None

db_url = None
try:
    if "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    elif "DATABASE" in st.secrets and "url" in st.secrets["DATABASE"]:
        db_url = st.secrets["DATABASE"]["url"]
except Exception:
    db_url = None

if not db_url:
    db_url = os.environ.get("DATABASE_URL")

if db_url:
    try:
        ENGINE = create_engine(db_url, pool_pre_ping=True)
        cols = [
            Column("id", Integer, primary_key=True),
            Column("submitted_at", SA_DateTime, nullable=False)
        ]
        for c in feature_cols:
            cols.append(Column(c.replace(" ", "_"), Float))

        cols += [
            Column("predicted_crop", String(256)),
            Column("predicted_proba", Text),
        ]

        submissions_table = Table("crop_submissions", metadata, *cols)
        metadata.create_all(ENGINE)
        db_status = "Connected to Cloud DB"
    except Exception as e:
        submissions_table = None
        db_status = f"DB failed: {str(e)}"
else:
    db_status = "No DB URL — submissions NOT saved"

# -----------------------------
# SAVE FUNCTION
# -----------------------------
def save_submission(conn, inputs, crop, proba):
    ins = {
        "submitted_at": datetime.utcnow(),
        "predicted_crop": crop,
        "predicted_proba": json.dumps(np.array(proba, dtype=float).tolist()) if proba is not None else None,
    }
    for k,v in inputs.items():
        ins[k] = float(v)

    conn.execute(submissions_table.insert().values(**ins))

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Smart Green Farm – Crop Recommendation")

with st.sidebar:
    st.info(db_status)
    st.subheader("Training Ranges")
    for col in feature_cols:
        mn, mx, _ = rng(col)
        st.write(f"{col}: {mn:.1f} → {mx:.1f}")

# -----------------------------
# INPUTS
# -----------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("Enter Conditions")

    def safe_slider(label, mn, mx, default, step=1.0):
        if np.isnan(mn) or np.isnan(mx) or mx <= mn:
            return st.number_input(label, value=default)
        default = float(max(min(default, mx), mn))
        return st.slider(label, float(mn), float(mx), default, float(step))

    vals = {}
    for col in feature_cols:
        mn, mx, mean = rng(col)
        vals[col] = safe_slider(col, mn, mx, mean)

    if st.button("Recommend Crop"):
        df_in = pd.DataFrame([vals], columns=feature_cols)
        crop = model.predict(df_in)[0]

        with right:
            st.success(f"🌱 Recommended Crop: **{crop}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_in)[0]
                st.write(pd.DataFrame({"Crop": model.classes_, "Prob": np.round(proba,3)}))

                if ENGINE and submissions_table is not None:
                    try:
                        with ENGINE.begin() as conn:
                            save_submission(conn, vals, crop, proba)
                        st.info("Submission saved!")
                    except Exception:
                        pass
