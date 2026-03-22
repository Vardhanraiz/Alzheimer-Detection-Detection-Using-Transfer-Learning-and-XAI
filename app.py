import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import base64
import io

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NeuroScan AI — Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# GLOBAL STYLES
# ======================================================
def inject_global_css():
    st.markdown("""
    <style>
    /* ── Import Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-deep:    #050b18;
        --bg-card:    rgba(10, 20, 40, 0.82);
        --accent:     #38bdf8;
        --accent2:    #818cf8;
        --danger:     #f87171;
        --success:    #34d399;
        --warn:       #fbbf24;
        --text-main:  #e2e8f0;
        --text-muted: #94a3b8;
        --border:     rgba(56, 189, 248, 0.18);
        --glow:       0 0 30px rgba(56,189,248,0.15);
    }

    /* ── Root & Body ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-main);
        background-color: var(--bg-deep);
    }

    /* ── Animated Gradient Background ── */
    .stApp {
        background:
            radial-gradient(ellipse 120% 80% at 10% 20%, rgba(56,189,248,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 80% 60% at 90% 80%, rgba(129,140,248,0.08) 0%, transparent 55%),
            linear-gradient(160deg, #050b18 0%, #080f22 50%, #060c1a 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* ── Brain network pattern overlay ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, rgba(56,189,248,0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08112a 0%, #060e22 100%) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

    /* ── Cards ── */
    .neuro-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: var(--glow), 0 4px 32px rgba(0,0,0,0.4);
        transition: box-shadow 0.3s ease;
    }
    .neuro-card:hover { box-shadow: 0 0 40px rgba(56,189,248,0.22), 0 4px 32px rgba(0,0,0,0.4); }

    /* ── Hero Header ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 6px 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 300;
        letter-spacing: 0.04em;
        margin: 0;
    }

    /* ── Section Headings ── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 0 0 20px 0;
    }

    /* ── Metric Cards ── */
    .metric-box {
        background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(129,140,248,0.06) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--accent); }

    /* ── Confidence Badge ── */
    .badge-high   { color: var(--success); border-color: var(--success); }
    .badge-medium { color: var(--warn);    border-color: var(--warn);    }
    .badge-low    { color: var(--danger);  border-color: var(--danger);  }
    .confidence-badge {
        display: inline-block;
        border: 1px solid;
        border-radius: 999px;
        padding: 3px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }

    /* ── Progress Bar Override ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
        border-radius: 999px;
    }
    .stProgress > div > div { background: rgba(255,255,255,0.06); border-radius: 999px; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.55rem 1.4rem;
        letter-spacing: 0.04em;
        transition: opacity 0.2s, transform 0.2s;
        box-shadow: 0 4px 20px rgba(56,189,248,0.2);
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
    .stButton > button:active { transform: translateY(0); }

    /* ── Logout button override ── */
    .logout-btn > button {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
        box-shadow: 0 4px 20px rgba(239,68,68,0.2) !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(56,189,248,0.04);
        border: 2px dashed var(--border);
        border-radius: 14px;
        padding: 12px;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--accent); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        color: var(--text-muted) !important;
        background: transparent !important;
        border: none !important;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--accent) !important;
        background: rgba(56,189,248,0.05) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
    }
    .streamlit-expanderContent { border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 8px 8px !important; }

    /* ── Alerts ── */
    .stAlert { border-radius: 12px !important; font-size: 0.9rem; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 24px 0; }

    /* ── Login Page ── */
    .login-wrapper {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .login-box {
        background: rgba(8, 17, 42, 0.92);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(56,189,248,0.22);
        border-radius: 24px;
        padding: 52px 48px;
        max-width: 440px;
        width: 100%;
        box-shadow: 0 0 60px rgba(56,189,248,0.1), 0 20px 80px rgba(0,0,0,0.5);
        animation: fadeUp 0.5s ease forwards;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .login-icon {
        font-size: 3.2rem;
        text-align: center;
        margin-bottom: 12px;
        animation: pulse 2.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50%       { transform: scale(1.06); }
    }
    .login-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    .login-subtitle {
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 32px;
        letter-spacing: 0.03em;
    }
    .login-input-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    /* ── Input styling ── */
    input[type="text"],
    input[type="password"],
    [data-testid="stTextInput"] input,
    .stTextInput input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(56,189,248,0.3) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        caret-color: #38bdf8 !important;
    }

    /* Input wrapper background */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        border-radius: 10px !important;
    }

    /* Remove inner border/outline duplication */
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.18) !important;
    }

    /* Placeholder text */
    input::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* Make sure text stays light after autofill */
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus {
        -webkit-text-fill-color: #f1f5f9 !important;
        -webkit-box-shadow: 0 0 0px 1000px rgba(15,28,55,0.95) inset !important;
        caret-color: #38bdf8 !important;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.75rem; font-weight: 600; color: var(--success);
        letter-spacing: 0.06em;
    }
    .status-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--success);
        animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* ── Stage Result Banner ── */
    .stage-banner {
        background: linear-gradient(135deg, rgba(56,189,248,0.1), rgba(129,140,248,0.08));
        border-left: 4px solid var(--accent);
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .stage-name { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; color: var(--accent); }
    .stage-desc { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }

    /* ── Disclaimer box ── */
    .disclaimer {
        background: rgba(251,191,36,0.06);
        border: 1px solid rgba(251,191,36,0.2);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.8rem;
        color: var(--warn);
        display: flex; align-items: flex-start; gap: 8px;
        margin-top: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_global_css()

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ======================================================
# CREDENTIALS (extend as needed)
# ======================================================
USERS = {
    "admin":    "neuro2026",
    "doctor":   "alz@2026",
    "research": "brain123",
}

# ======================================================
# LOGIN PAGE
# ======================================================
def show_login():
    # Centered login card via columns
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown("""
        <div class="login-box">
            <div class="login-icon">🧠</div>
            <div class="login-title">NeuroScan AI</div>
            <div class="login-subtitle">Explainable AI · Alzheimer's Detection Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-input-label">Username</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed", key="login_user")

        st.markdown('<div class="login-input-label" style="margin-top:14px;">Password</div>', unsafe_allow_html=True)
        password = st.text_input("Password", placeholder="Enter your password", type="password", label_visibility="collapsed", key="login_pass")

        st.write("")
        if st.button("🔐  Sign In", use_container_width=True):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌  Invalid credentials. Please try again.")

        st.markdown("""
        <div class="disclaimer">
            <span>⚠️</span>
            <span>This platform is for <strong>academic & research use only</strong>. 
            Not intended for clinical diagnosis or medical decision-making.</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><p style='text-align:center;font-size:0.75rem;color:#475569;'>Demo credentials — user: <code>admin</code> · pass: <code>neuro2026</code></p>", unsafe_allow_html=True)


# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("phase1_best.keras
")

# ======================================================
# CLASSES & DESCRIPTIONS
# ======================================================
class_names = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

class_descriptions = {
    "Non-Demented":        "No signs of cognitive decline detected. Brain structures appear within normal range.",
    "Very Mild Demented":  "Very early-stage indicators present. Subtle changes in memory-related regions.",
    "Mild Demented":       "Mild atrophy observed. Hippocampal and cortical changes consistent with MCI.",
    "Moderate Demented":   "Moderate neurodegeneration detected. Significant structural changes are visible."
}

class_colors = {
    "Non-Demented":        "#34d399",
    "Very Mild Demented":  "#fbbf24",
    "Mild Demented":       "#fb923c",
    "Moderate Demented":   "#f87171"
}

# ======================================================
# IMAGE PROCESSING UTILITIES
# ======================================================
def is_likely_mri(img: Image.Image) -> bool:
    img_gray = np.array(img.convert("L"))
    mean_intensity = img_gray.mean()
    std_intensity  = img_gray.std()
    return mean_intensity < 200 and std_intensity > 10

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================================================
# GRAD-CAM
# ======================================================
def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str = "top_conv") -> np.ndarray:
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = tf.reduce_max(predictions, axis=1)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def overlay_gradcam(img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_rgb   = np.array(img.convert("RGB").resize((224, 224))).astype("float32")
    overlay   = heatmap_colored * 0.65 + img_rgb * 0.6
    return np.clip(overlay, 0, 255).astype("uint8")

# ======================================================
# LIME
# ======================================================
def lime_predict(images: np.ndarray) -> np.ndarray:
    return load_model().predict(np.array(images) / 255.0)

def generate_lime_explanation(img: Image.Image) -> np.ndarray:
    explainer  = lime_image.LimeImageExplainer()
    img_array  = np.array(img.convert("RGB").resize((224, 224)))
    explanation = explainer.explain_instance(
        img_array, lime_predict,
        top_labels=1, hide_color=0, num_samples=300
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True, num_features=5, hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask)

# ======================================================
# SIDEBAR (logged-in)
# ======================================================
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 4px 0 20px 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                        background:linear-gradient(135deg,#38bdf8,#818cf8);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">
                🧠 NeuroScan AI
            </div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">Alzheimer's Detection Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="status-pill">
            <div class="status-dot"></div>
            Logged in as <strong>&nbsp;{st.session_state.username}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 📋 About")
        st.markdown("""
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.7;">
        <b style="color:#e2e8f0;">Approach:</b> Transfer Learning + XAI<br>
        <b style="color:#e2e8f0;">Explainability:</b> Grad-CAM · LIME<br>
        <b style="color:#e2e8f0;">Architecture:</b> EfficientNet-B0<br>
        <b style="color:#e2e8f0;">Input:</b> Axial Brain MRI (224×224)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Dataset")
        st.markdown("""
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
        <b style="color:#e2e8f0;">Classes:</b><br>
        <span style="color:#34d399;">●</span> Non-Demented<br>
        <span style="color:#fbbf24;">●</span> Very Mild Demented<br>
        <span style="color:#fb923c;">●</span> Mild Demented<br>
        <span style="color:#f87171;">●</span> Moderate Demented
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚙️ Settings")
        num_lime_samples = st.slider("LIME Samples", 100, 500, 300, 50,
                                     help="More samples = better explanation, slower computation")
        num_lime_features = st.slider("LIME Features", 3, 10, 5, 1,
                                      help="Number of superpixel regions to highlight")

        st.markdown("<br>", unsafe_allow_html=True)

        # Logout
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 NeuroScan AI · Academic Use Only")

    return num_lime_samples, num_lime_features

# ======================================================
# MAIN APP
# ======================================================
def show_app():
    num_lime_samples, num_lime_features = show_sidebar()

    # ── Header ──────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown('<p class="hero-title">Alzheimer\'s MRI Analysis</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Explainable AI-powered brain scan interpretation · For research & academic use</p>', unsafe_allow_html=True)
    with col_badge:
        st.markdown("""
        <div style="text-align:right;padding-top:10px;">
            <div class="status-pill">
                <div class="status-dot"></div> Model Ready
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        # Upload section
        st.markdown('<p class="section-label">Step 01</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Upload Brain MRI</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="neuro-card" style="padding:18px 20px;margin-bottom:12px;">
        <span style="font-size:0.82rem;color:#94a3b8;">
        📌 Upload a <b style="color:#e2e8f0;">single axial brain MRI</b> image (JPG / PNG).<br><br>
        Best results with clear, centered, grayscale MRI slices. Avoid photos of printouts or colored scans.
        </span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button("🔍  Run Analysis", use_container_width=True)
        else:
            analyze_clicked = False
            # Placeholder instructions
            st.markdown("""
            <div style="border:2px dashed rgba(56,189,248,0.15);border-radius:12px;
                        padding:40px 20px;text-align:center;color:#475569;font-size:0.85rem;margin-top:8px;">
                🧠<br><br>No MRI uploaded yet.<br>Upload an image to begin analysis.
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        if uploaded_file and analyze_clicked:
            img = Image.open(uploaded_file)

            if not is_likely_mri(img):
                st.error("❌ The image doesn't appear to be a brain MRI. Please upload a valid grayscale axial MRI scan.")
                st.stop()

            with st.spinner("🧠  Analyzing scan — please wait…"):
                model = load_model()
                processed_img = preprocess_image(img)
                preds         = model.predict(processed_img)
                pred_idx      = int(np.argmax(preds))
                pred_class    = class_names[pred_idx]
                confidence    = float(np.max(preds))

            # ── Prediction Summary ───────────────────────
            st.markdown('<p class="section-label">Step 02</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Prediction Summary</p>', unsafe_allow_html=True)

            # Stage banner
            stage_color = class_colors[pred_class]
            st.markdown(f"""
            <div class="stage-banner" style="border-left-color:{stage_color};">
                <div class="stage-name" style="color:{stage_color};">{pred_class}</div>
                <div class="stage-desc">{class_descriptions[pred_class]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metric row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="color:{stage_color};">{confidence*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Stage Index</div>
                    <div class="metric-value">{pred_idx + 1} / 4</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                severity = ["None", "Low", "Moderate", "High"][pred_idx]
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Severity</div>
                    <div class="metric-value" style="color:{stage_color};">{severity}</div>
                </div>""", unsafe_allow_html=True)

            # Class probabilities
            with st.expander("📊 Full Class Probabilities"):
                for i, cls in enumerate(class_names):
                    col_a, col_b = st.columns([3, 1])
                    prob = float(preds[0][i])
                    with col_a:
                        st.markdown(f"<span style='font-size:0.82rem;color:#94a3b8;'>{cls}</span>", unsafe_allow_html=True)
                        st.progress(prob)
                    with col_b:
                        st.markdown(f"<div style='padding-top:24px;font-size:0.88rem;font-weight:600;"
                                    f"color:{class_colors[cls]};text-align:right;'>{prob*100:.1f}%</div>",
                                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Explainability ───────────────────────────
            st.markdown('<p class="section-label">Step 03</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Explainable AI Visualizations</p>', unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["🌡️  Grad-CAM Heatmap", "🔬  LIME Explanation"])

            with tab1:
                with st.spinner("Generating Grad-CAM…"):
                    heatmap = make_gradcam_heatmap(processed_img, model)
                    superimposed = overlay_gradcam(img, heatmap)
                st.image(superimposed, use_column_width=True)
                st.markdown("""
                <div style="font-size:0.8rem;color:#64748b;text-align:center;margin-top:6px;">
                    Grad-CAM · Warmer colors indicate regions most influential to the prediction
                </div>""", unsafe_allow_html=True)

            with tab2:
                with st.spinner("Generating LIME explanation (this may take ~30s)…"):
                    lime_result = generate_lime_explanation(img)
                st.image(lime_result, use_column_width=True)
                st.markdown("""
                <div style="font-size:0.8rem;color:#64748b;text-align:center;margin-top:6px;">
                    LIME · Highlighted superpixels contribute most to the predicted class
                </div>""", unsafe_allow_html=True)

            # ── Disclaimer ───────────────────────────────
            st.markdown("""
            <div class="disclaimer">
                <span>⚠️</span>
                <span>This analysis is generated by an AI model and is intended solely for
                <strong>research and educational purposes</strong>. It must not be used for clinical
                diagnosis, treatment planning, or any medical decisions.</span>
            </div>
            """, unsafe_allow_html=True)

        elif not uploaded_file:
            # Idle state placeholder
            st.markdown("""
            <div class="neuro-card" style="min-height:420px;display:flex;flex-direction:column;
                         align-items:center;justify-content:center;gap:16px;">
                <div style="font-size:4rem;opacity:0.3;">🧠</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                             color:#334155;text-align:center;">
                    Results will appear here
                </div>
                <div style="font-size:0.82rem;color:#475569;text-align:center;max-width:280px;line-height:1.6;">
                    Upload a brain MRI image and click<br><strong style="color:#64748b;">Run Analysis</strong> to get started.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────
    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.caption("© 2026 NeuroScan AI · Academic Project")
    with fc2:
        st.caption("🧠 Explainable AI for Alzheimer's Disease")
    with fc3:
        st.caption("For research use only · Not for clinical diagnosis")

# ======================================================
# ENTRY POINT
# ======================================================
if not st.session_state.logged_in:
    show_login()
else:
    show_app()
