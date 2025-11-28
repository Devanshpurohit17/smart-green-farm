import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Green Farm – Crop Recommendation",
    page_icon="🌿",
    layout="wide",
)

# ---------------------------------------------------
# CUSTOM CSS (only for Model A)
# ---------------------------------------------------
custom_css = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #e0f7fa 0%, #f1f8e9 40%, #e8f5e9 100%);
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0b3d1a;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 15px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: 0.03em;
        animation: fadeInUp 0.5s ease-out;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #1b5e20;
        opacity: 0.85;
        margin-bottom: 18px;
        animation: fadeInUp 0.7s ease-out;
    }

    .glass-card {
        background: rgba(255,255,255,0.96);
        border-radius: 18px;
        padding: 18px 20px 16px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.6);
        animation: fadeInUp 0.6s ease-out;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 55px rgba(0,0,0,0.12);
        transition: 0.2s ease-out;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .metric-card {
        background: linear-gradient(135deg, #a5d6a7, #66bb6a);
        border-radius: 16px;
        padding: 14px;
        color: #0b2e13 !important;
        font-weight: 600;
        text-align: left;
        box-shadow: 0 14px 30px rgba(56,142,60,0.45);
        margin-bottom: 6px;
    }

    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        opacity: 0.8;
    }

    .metric-value {
        font-size: 22px;
        margin-top: 4px;
    }

    /* Inputs clearly visible */
    label, .stMarkdown, .stText {
        color: #0b3d1a !important;
    }

    .stNumberInput input, .stTextInput input {
        background-color: #ffffff !important;
        color: #0b3d1a !important;
        border-radius: 10px;
        border: 1px solid #c8e6c9;
    }

    .stSlider label, .stSlider span {
        color: #0b3d1a !important;
    }

    .block-container {
        padding-top: 1.2rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------
DATA_PATH = r"E:\AIML ASSIGNMENT\Crop_recommendation.csv"
MODEL_PATH = r"E:\AIML ASSIGNMENT\hybrid_crop_reco_model.pkl"
FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ---------------------------------------------------
# LOAD MODEL + DATA
# ---------------------------------------------------
@st.cache_resource
def load_model_and_stats():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    stats = df[FEATURE_COLS].describe()
    return model, stats

model, stats = load_model_and_stats()

def rng(col):
    """Return (min, max, mean) for slider ranges."""
    return (
        float(stats.loc["min", col]),
        float(stats.loc["max", col]),
        float(stats.loc["mean", col]),
    )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 Smart Green Farm – Model A")
    st.write(
        "This interface uses a trained machine learning model "
        "to recommend the most suitable crop based on soil fertility "
        "and weather conditions."
    )

    st.markdown("---")
    st.write("**Training Data Ranges:**")
    for col in FEATURE_COLS:
        mn, mx, _ = rng(col)
        st.write(f"- {col}: {mn:.1f} – {mx:.1f}")
    st.markdown("---")
    st.caption(
        "For realistic predictions, keep your inputs inside these ranges."
    )

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown('<div class="hero-title">Smart Green Farm – Crop Recommendation</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Adjust your field conditions and get an instant crop suggestion.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------
left, right = st.columns([2, 1])

# ---------------- LEFT: INPUTS ----------------------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter Soil & Weather Parameters</div>',
                unsafe_allow_html=True)

    N_min, N_max, N_mean = rng("N")
    P_min, P_max, P_mean = rng("P")
    K_min, K_max, K_mean = rng("K")
    T_min, T_max, T_mean = rng("temperature")
    H_min, H_max, H_mean = rng("humidity")
    pH_min, pH_max, pH_mean = rng("ph")
    R_min, R_max, R_mean = rng("rainfall")

    N = st.slider("Nitrogen (N)", int(N_min), int(N_max), int(N_mean), step=1)
    P = st.slider("Phosphorus (P)", int(P_min), int(P_max), int(P_mean), step=1)
    K = st.slider("Potassium (K)", int(K_min), int(K_max), int(K_mean), step=1)

    temperature = st.slider("Temperature (°C)", float(T_min), float(T_max),
                            float(T_mean), step=0.5)
    humidity = st.slider("Humidity (%)", float(H_min), float(H_max),
                         float(H_mean), step=1.0)
    ph = st.slider("Soil pH", float(pH_min), float(pH_max),
                   float(pH_mean), step=0.1)
    rainfall = st.slider("Rainfall (mm)", float(R_min), float(R_max),
                         float(R_mean), step=1.0)

    btn = st.button("🔍 Recommend Crop", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT: OUTPUT ---------------------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Output</div>',
                unsafe_allow_html=True)

    if btn:
        input_df = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=FEATURE_COLS,
        )
        pred_crop = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Crop</div>
                <div class="metric-value">{pred_crop}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "The model suggests this crop based on multi-class classification "
            "trained on historical soil–crop data."
        )
    else:
        st.write("Set the parameters on the left and click **Recommend Crop**.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Smart Green Farm – AIML Project | Model A (Crop Recommendation using 4 supervised + 2 unsupervised algorithms)")
