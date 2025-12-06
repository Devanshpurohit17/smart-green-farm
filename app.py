# app.py

import json
from pathlib import Path
from datetime import datetime
import sqlite3

import joblib
import pandas as pd
import numpy as np
import streamlit as st

# =========================================================
# BASIC CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILENAME = "Crop_recommendation.csv"
MODEL_FILENAME = "hybrid_yield_model.pkl"   # <-- YOUR ACTUAL MODEL FILE

DATA_PATH = BASE_DIR / DATA_FILENAME
MODEL_PATH = BASE_DIR / MODEL_FILENAME

st.set_page_config(
    page_title="Smart Green Farm - Crop Recommendation",
    page_icon="🌿",
    layout="wide",
)

# --------- Small CSS touch to make it look premium ----------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top left, #182848, #000000 60%);
        color: #f5f5f5;
    }

    /* Title styling */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subheading */
    .subtitle {
        color: #d1d5db;
        font-size: 16px;
        margin-bottom: 1rem;
    }

    /* Card style */
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 18px 45px rgba(0,0,0,0.55);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
    }

    /* Sliders text color fix */
    .stSlider label, .stNumberInput label {
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL + DATA LOAD
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except EOFError:
        st.error(
            f"Model file '{MODEL_PATH.name}' seems to be empty or corrupted.\n"
            "Please regenerate and replace it with a valid .pkl file."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset missing: {DATA_PATH.name}")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for col in feature_cols:
        if col not in data.columns:
            st.error(f"Feature '{col}' missing in CSV")
            st.stop()

    stats = data[feature_cols].describe()

    if "min" not in stats.index or "max" not in stats.index:
        st.error("CSV is malformed — stats not available")
        st.stop()

    # Figure out crop label column if present
    crop_col = None
    for cand in ["label", "crop", "Crop"]:
        if cand in data.columns:
            crop_col = cand
            break

    return data, stats, feature_cols, crop_col


model = load_model()
full_data, stats, feature_cols, crop_col = load_data()


def rng(col):
    mn = float(stats.loc["min", col])
    mx = float(stats.loc["max", col])
    mean = float(stats.loc["mean", col])
    return mn, mx, mean

# =========================================================
# DATABASE (SQLite local file, using sqlite3 ONLY)
# =========================================================
DB_PATH = BASE_DIR / "crop_submissions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crop_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT NOT NULL,
            N REAL,
            P REAL,
            K REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            predicted_crop TEXT,
            predicted_proba TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Try to init DB and set a nice status message
try:
    init_db()
    db_status = f"✅ Local DB connected (SQLite file: {DB_PATH.name})"
except Exception as e:
    db_status = f"⚠️ DB error: {e}"


def save_submission(inputs, crop, proba):
    """Insert a row into crop_submissions table using sqlite3."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO crop_submissions
            (submitted_at, N, P, K, temperature, humidity, ph, rainfall,
             predicted_crop, predicted_proba)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                float(inputs["N"]),
                float(inputs["P"]),
                float(inputs["K"]),
                float(inputs["temperature"]),
                float(inputs["humidity"]),
                float(inputs["ph"]),
                float(inputs["rainfall"]),
                crop,
                json.dumps(np.array(proba, dtype=float).tolist())
                if proba is not None
                else None,
            ),
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🌾 System Status")
    st.info(db_status)

    st.markdown("### 📊 Training Ranges")
    for col in feature_cols:
        mn, mx, _ = rng(col)
        st.write(f"**{col}**: {mn:.1f} → {mx:.1f}")

    if crop_col:
        st.markdown("---")
        st.markdown("### 🧬 Dataset Snapshot")
        total_rows = len(full_data)
        total_crops = full_data[crop_col].nunique()
        st.metric("Total Samples", total_rows)
        st.metric("Unique Crops", total_crops)

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Smart Green Farm – Crop Recommendation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'AI-powered assistant to help farmers choose the most suitable crop '
    'based on soil and weather conditions.'
    '</div>',
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT: INPUTS (LEFT)  |  RESULTS (RIGHT)
# =========================================================
left_col, right_col = st.columns([1.9, 1.4], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🧪 Enter Field Conditions")

    def safe_slider(label, mn, mx, default, step=1.0):
        if np.isnan(mn) or np.isnan(mx) or mx <= mn:
            return st.number_input(label, value=default)
        default = float(max(min(default, mx), mn))
        return st.slider(label, float(mn), float(mx), default, float(step))

    vals = {}
    for col in feature_cols:
        mn, mx, mean = rng(col)
        vals[col] = safe_slider(col, mn, mx, mean)

    predict_clicked = st.button("🌱 Recommend Crop", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

results_container = right_col.container()

# =========================================================
# PREDICTION + DISPLAY
# =========================================================
if predict_clicked:
    df_in = pd.DataFrame([vals], columns=feature_cols)
    crop = model.predict(df_in)[0]

    with results_container:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("✅ Recommendation Result")

        st.success(f"🌿 **Recommended Crop: `{crop}`**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_in)[0]

            prob_df = pd.DataFrame(
                {"Crop": model.classes_, "Probability": np.round(proba, 3)}
            ).sort_values("Probability", ascending=False)

            st.markdown("#### 📈 Confidence Scores")
            st.dataframe(
                prob_df.reset_index(drop=True),
                use_container_width=True,
                height=350,
            )
        else:
            proba = None
            st.info("Model does not provide probability scores.")

        # Save to DB
        ok, err = save_submission(
            inputs={
                "N": vals["N"],
                "P": vals["P"],
                "K": vals["K"],
                "temperature": vals["temperature"],
                "humidity": vals["humidity"],
                "ph": vals["ph"],
                "rainfall": vals["rainfall"],
            },
            crop=crop,
            proba=proba,
        )

        if ok:
            st.success("🗄️ This prediction has been saved in the local database.")
        else:
            st.warning(f"Could not save to database: {err}")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# VIEW SAVED DATA
# =========================================================
st.markdown("## 📁 Saved Submissions (Database View)")
with st.expander("Show all saved records"):
    try:
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            df_saved = pd.read_sql_query(
                "SELECT * FROM crop_submissions ORDER BY id DESC", conn
            )
            conn.close()

            if not df_saved.empty:
                st.dataframe(df_saved, use_container_width=True, height=400)
            else:
                st.info(
                    "No records found yet. Submit a prediction to populate the table."
                )
        else:
            st.info("Database file not created yet. Submit a prediction first.")
    except Exception as e:
        st.error(f"Error reading from database: {e}")

