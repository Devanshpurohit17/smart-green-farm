# app.py
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st

# SQLAlchemy
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Text, DateTime as SA_DateTime
)
from sqlalchemy.exc import SQLAlchemyError

# -------------------------
# CONFIG / PATHS
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Stress_Dataset.csv"
MODEL_PATH = BASE_DIR / "best_stress_model.pkl"
TARGET_COL = "Which type of stress do you primarily experience?"

# Streamlit page config
st.set_page_config(page_title="PsyTrack – Student Stress Analyzer", page_icon="🧠", layout="wide")

# -------------------------
# CSS / Theme (short)
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: #08080b; color: #e6eef8; }
    .hero-title { font-size:24px; font-weight:800; color:#fff; margin-top:18px; }
    .card { background: rgba(20,24,31,0.9); border-radius:12px; padding:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HELPER: load model / data
# -------------------------
@st.cache_resource
def load_model_and_data():
    if not DATA_PATH.exists():
        st.error(f"Missing dataset file: {DATA_PATH.name}. Put it next to app.py in repo.")
        st.stop()
    if not MODEL_PATH.exists():
        st.error(f"Missing model file: {MODEL_PATH.name}. Put it next to app.py in repo.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found in the CSV. Fix CSV or TARGET_COL constant.")
        st.stop()

    # Keep only numeric features for sliders
    X = df.drop(columns=[TARGET_COL])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric features found in dataset.")
        st.stop()

    stats = X[numeric_cols].describe()
    model = joblib.load(MODEL_PATH)
    return model, df, numeric_cols, stats

model, full_df, feature_cols, stats = load_model_and_data()

def feature_range(col):
    mn = float(stats.loc["min", col])
    mx = float(stats.loc["max", col])
    mean_val = float(stats.loc["mean", col])
    return mn, mx, mean_val

# -------------------------
# DATABASE: connect (Supabase/Heroku) via st.secrets or ENV
# -------------------------
ENGINE = None
metadata = MetaData()
submissions_table = None

db_url = None
# First check Streamlit secrets (recommended)
try:
    # If user saved DATABASE_URL directly in streamlit secrets root:
    if "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    # Older pattern: nested dictionary, optional
    elif "DATABASE" in st.secrets and "url" in st.secrets["DATABASE"]:
        db_url = st.secrets["DATABASE"]["url"]
except Exception:
    # no secrets set or not accessible
    db_url = None

# Fallback to environment variable (local testing only)
if not db_url:
    import os
    db_url = os.environ.get("DATABASE_URL")

if db_url:
    try:
        ENGINE = create_engine(db_url, pool_pre_ping=True)
        # Define table schema dynamically (one float column for each numeric feature)
        cols = [
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("submitted_at", SA_DateTime, nullable=False),
            Column("user_id", String(128), nullable=True),
            Column("source", String(64), nullable=True),
        ]
        # Add feature columns (Float)
        for c in feature_cols:
            safe_name = c.replace(" ", "_").replace("-", "_")[:60]
            cols.append(Column(safe_name, Float))

        # Add prediction/metadata columns
        cols += [
            Column("predicted_label", String(256)),
            Column("predicted_proba", Text),  # JSON string
            Column("notes", Text, nullable=True)
        ]

        submissions_table = Table("submissions", metadata, *cols)
        metadata.create_all(ENGINE)
        st.sidebar.success("🔒 Connected to cloud DB (Postgres). Submissions will be saved.")
    except SQLAlchemyError as e:
        st.sidebar.error("DB connection failed. Check DATABASE_URL in Streamlit secrets.")
        st.sidebar.write(str(e))
        ENGINE = None
else:
    st.sidebar.warning("No DATABASE_URL found in Streamlit secrets or environment. Submissions disabled.")

# -------------------------
# Helper: save submission
# -------------------------
def save_submission(conn, input_dict, pred_label=None, proba=None, user_id=None, source="streamlit_app", notes=None):
    """
    conn: engine.connect() or engine.begin()
    input_dict: original numeric inputs keyed by feature_cols
    proba: numpy array or list (will be json-dumped)
    """
    # Build insert dict: convert feature names to safe column names used above
    insert_dict = {
        "submitted_at": datetime.utcnow(),
        "user_id": user_id,
        "source": source,
        "predicted_label": str(pred_label) if pred_label is not None else None,
        "predicted_proba": json.dumps(proba.tolist() if hasattr(proba, "tolist") else proba) if proba is not None else None,
        "notes": notes
    }
    for k, v in input_dict.items():
        safe_name = k.replace(" ", "_").replace("-", "_")[:60]
        try:
            insert_dict[safe_name] = float(v) if v is not None else None
        except Exception:
            insert_dict[safe_name] = None

    ins = submissions_table.insert().values(**insert_dict)
    res = conn.execute(ins)
    # Return primary key if available
    try:
        pk = res.inserted_primary_key[0]
        return pk
    except Exception:
        return None

# -------------------------
# UI: Sidebar + Hero
# -------------------------
with st.sidebar:
    st.title("PsyTrack Dashboard")
    st.write("AI Student Stress Analyzer — Model A")
    st.markdown("---")
    st.write("**Feature ranges (from training data):**")
    for c in feature_cols:
        mn, mx, _ = feature_range(c)
        st.write(f"- {c} → `{mn:.1f} — {mx:.1f}`")
    st.markdown("---")
    if ENGINE:
        st.write("DB: Connected ✅")
    else:
        st.write("DB: Not configured — submissions will not be saved.")

st.markdown('<div class="hero-title">PsyTrack – Student Stress Analyzer</div>', unsafe_allow_html=True)
st.write("Adjust the questionnaire sliders and click **Analyze** to predict stress type.")

st.markdown("---")

# -------------------------
# Main layout (inputs & outputs)
# -------------------------
left, right = st.columns([2.2, 1])
inputs = {}

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Questionnaire Responses")

    colA, colB = st.columns(2)
    for i, col in enumerate(feature_cols):
        mn, mx, mean_val = feature_range(col)

        # Guard: if min==max -> use disabled number input
        if mx <= mn:
            box = colA if i % 2 == 0 else colB
            with box:
                st.info(f"'{col}' is constant (value={mn}).")
                st.number_input(col, value=float(mn), disabled=True)
            inputs[col] = float(mn)
            continue

        is_int = mn.is_integer() and mx.is_integer() and (mx - mn <= 10)
        default = min(max(mean_val, mn), mx)
        if is_int:
            default = float(int(round(default)))
            step = 1.0
        else:
            rng = mx - mn
            step = max(min(rng / 40.0, rng / 2.0), 0.01)

        box = colA if i % 2 == 0 else colB
        with box:
            val = st.slider(col, min_value=float(mn), max_value=float(mx), value=float(default), step=float(step))
            inputs[col] = int(round(val)) if is_int else float(val)

    analyze = st.button("Analyze Stress Type", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analysis Result")

    if analyze:
        # Prepare input for prediction
        df_input = pd.DataFrame([inputs], columns=feature_cols)
        try:
            pred = model.predict(df_input)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred = None

        st.markdown(f"**Predicted stress type:**  \n> {pred}")

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_input)[0]
                prob_df = pd.DataFrame({"Stress Type": model.classes_, "Probability": np.round(proba, 3)})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                st.markdown("**Prediction confidence**")
                st.table(prob_df.reset_index(drop=True))
            except Exception as e:
                st.info("Probability unavailable: " + str(e))

        # Save to DB if engine available
        if ENGINE and submissions_table is not None:
            try:
                with ENGINE.begin() as conn:
                    newid = save_submission(conn, inputs, pred_label=pred, proba=proba, user_id=None, source="streamlit_cloud")
                st.success(f"✅ Submission saved to DB (id={newid})")
            except Exception as e:
                st.error(f"Failed to save submission to DB: {e}")
        else:
            st.info("DB not configured — submission not saved.")

    else:
        st.write("Set slider values and click **Analyze Stress Type**.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Admin: optional preview (only if DB connected)
# -------------------------
if ENGINE:
    st.markdown("---")
    st.subheader("Recent submissions (DB preview)")
    try:
        # Simple preview: read last 5 rows
        query = submissions_table.select().order_by(submissions_table.c.id.desc()).limit(5)
        with ENGINE.connect() as conn:
            res = conn.execute(query)
            rows = [dict(r) for r in res]
        if rows:
            preview_df = pd.DataFrame(rows)
            st.dataframe(preview_df)
        else:
            st.write("No submissions yet.")
    except Exception as e:
        st.info("Could not load submissions preview: " + str(e))
