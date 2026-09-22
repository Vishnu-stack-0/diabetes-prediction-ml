import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import math

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "src" / "logo.png"

# -------------------------------------------------
# 1. Page config MUST be first
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# 2. Loading Screen
# -------------------------------------------------
def show_loading_screen(logo_path: Path, duration_ms: int = 2000):
    if not logo_path.exists():
        return
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    #splash-screen {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f8fafc 100%);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        z-index: 999999; animation: fadeOutSplash 0.7s ease-out {duration_ms}ms forwards;
    }}
    .splash-logo {{
        width: 120px; height: 120px; border-radius: 50%; object-fit: cover;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25); border: 4px solid white;
        animation: logoPulse 1.4s ease-in-out infinite;
    }}
    .splash-spinner {{
        margin-top: 24px; width: 36px; height: 36px;
        border: 3.5px solid #bae6fd; border-top: 3.5px solid #0ea5e9;
        border-radius: 50%; animation: spin 0.85s linear infinite;
    }}
    .splash-text {{ margin-top: 16px; font-size: 0.95rem; color: #0369a1; font-weight: 500; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes logoPulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
    @keyframes fadeOutSplash {{ to {{ opacity: 0; visibility: hidden; pointer-events: none; }} }}
    </style>
    <div id="splash-screen">
        <img src="data:image/png;base64,{logo_b64}" class="splash-logo" alt="Logo">
        <div class="splash-spinner"></div>
        <div class="splash-text">Loading Diabetes Risk Predictor...</div>
    </div>
    """, unsafe_allow_html=True)

show_loading_screen(LOGO_PATH)

# -------------------------------------------------
# 3. Background CSS
# -------------------------------------------------
def set_background(image_path: Path, opacity: float = 0.11):
    if not image_path.exists():
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255,255,255,{1-opacity}), rgba(248,250,252,{1-opacity})),
                          url("data:image/png;base64,{encoded}");
        background-size: cover, 220px;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .stApp, .stMarkdown, .stText, p, span, label, .stCaption, div, li {{ color: #1e293b !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: #0f172a !important; font-weight: 700 !important; }}
    div[data-testid="stForm"] {{
        background-color: rgba(255,255,255,0.95) !important;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    }}
    .stNumberInput > div > div > input, .stSelectbox > div > div {{
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }}
    .stFormSubmitButton > button {{
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        width: 100%;
    }}
    .stProgress > div > div > div > div {{ background-color: #0ea5e9 !important; }}
    .streamlit-expanderHeader {{ background-color: rgba(241,245,249,0.95) !important; border-radius: 10px !important; }}
    .streamlit-expanderContent {{ background-color: rgba(255,255,255,0.97) !important; }}
    .stSelectbox > div > div, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: #ffffff !important; color: #1e293b !important;
    }}
    div[data-baseweb="popover"], div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {{
        background-color: #ffffff !important; color: #1e293b !important;
    }}
    div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {{
        background-color: #e0f2fe !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background(LOGO_PATH)

# -------------------------------------------------
# 4. Model loading
# -------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "diabetes_pipeline.joblib"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()
preproc = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# -------------------------------------------------
# 5. UI
# -------------------------------------------------
st.title("Diabetes Risk Prediction")
st.caption("Enter patient details to estimate diabetes risk.")

with st.expander("📋 What do these numbers mean?"):
    st.markdown("""
| Field         | Normal range     | Notes                |
|---------------|------------------|----------------------|
| Age           | any adult        | years                |
| Pulse rate    | 60–100 bpm       | resting              |
| Systolic BP   | 90–120 mmHg      | top number           |
| Diastolic BP  | 60–80 mmHg       | bottom number        |
| Glucose       | 4–6 mmol/L       | fasting              |
| BMI           | 18.5–24.9 normal | from height & weight |
| Binary fields | 0 = No, 1 = Yes  | medical history      |
""")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 1, 120, 45)
        gender = st.selectbox("Gender", ["Female", "Male"])
        pulse_rate = st.number_input("Pulse rate (bpm)", 30, 200, 76)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 220, 130)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 130, 81)
        glucose = st.number_input("Fasting glucose (mmol/L)", 0.0, 40.0, 6.9, 0.1)
        height = st.number_input("Height (metres)", 1.0, 2.2, 1.55, 0.01)
    with col2:
        weight = st.number_input("Weight (kg)", 20.0, 200.0, 58.0, 0.5)
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        family_diabetes = st.selectbox("Family history of diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        hypertensive = st.selectbox("Currently hypertensive", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        family_hypertension = st.selectbox("Family history of hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        cardiovascular_disease = st.selectbox("Cardiovascular disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        stroke = st.selectbox("History of stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    submitted = st.form_submit_button("Predict")

# -------------------------------------------------
# 6. Prediction + SHAP + Architecture Simulation
# -------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "age": age, "gender": gender, "pulse_rate": pulse_rate,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "glucose": glucose, "height": height, "weight": weight, "bmi": bmi,
        "family_diabetes": family_diabetes, "hypertensive": hypertensive,
        "family_hypertension": family_hypertension,
        "cardiovascular_disease": cardiovascular_disease, "stroke": stroke
    }])

    prob = float(pipeline.predict_proba(input_df)[0, 1])
    pred = int(pipeline.predict(input_df)[0])

    eps = 1e-9
    p_clamped = min(max(prob, eps), 1 - eps)
    raw_score = math.log(p_clamped / (1 - p_clamped))

    st.divider()
    st.subheader("Result")
    if pred == 1:
        st.error(f"Higher diabetes risk — estimated probability: {prob:.1%}")
    else:
        st.success(f"Lower diabetes risk — estimated probability: {prob:.1%}")
    st.progress(min(int(prob * 100), 100))

    flags = []
    if glucose >= 7:
        flags.append("Fasting glucose is in the diabetic range (≥7 mmol/L).")
    elif glucose >= 6:
        flags.append("Fasting glucose is in the pre-diabetic range (6–7 mmol/L).")
    if bmi >= 30:
        flags.append("BMI is in the obese range (30+).")
    elif bmi >= 25:
        flags.append("BMI is in the overweight range (25–29.9).")
    if systolic_bp >= 140 or diastolic_bp >= 90:
        flags.append("Blood pressure is in the high range (≥140/90 mmHg).")
    if flags:
        st.warning("Values outside the normal range:\n\n" + "\n".join(f"- {f}" for f in flags))

    # SHAP
    st.divider()
    st.subheader("Why this prediction?")
    input_transformed = preproc.transform(input_df)
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()
    feature_names = preproc.get_feature_names_out()

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(input_transformed)
    base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]
    sample_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=input_transformed[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(sample_explanation, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # -------------------------------------------------
    # Model Architecture Diagram + Interactive Simulation
    # -------------------------------------------------
    st.divider()
    st.subheader("Model Architecture: XGBoost Pipeline")

    n_estimators = getattr(model, "n_estimators", "N/A")
    max_depth = getattr(model, "max_depth", "N/A")
    learning_rate = getattr(model, "learning_rate", "N/A")

    all_features = [
        ("Age", f"{age}"),
        ("Gender", gender),
        ("Pulse Rate", f"{pulse_rate} bpm"),
        ("Systolic BP", f"{systolic_bp} mmHg"),
        ("Diastolic BP", f"{diastolic_bp} mmHg"),
        ("Glucose", f"{glucose:.1f} mmol/L"),
        ("Height", f"{height:.2f} m"),
        ("Weight", f"{weight:.1f} kg"),
        ("BMI", f"{bmi:.1f}"),
        ("Family Diabetes", "Yes" if family_diabetes == 1 else "No"),
        ("Hypertensive", "Yes" if hypertensive == 1 else "No"),
        ("Family Hypertension", "Yes" if family_hypertension == 1 else "No"),
        ("Cardiovascular Disease", "Yes" if cardiovascular_disease == 1 else "No"),
        ("Stroke History", "Yes" if stroke == 1 else "No"),
    ]
    feature_grid_html = "".join(
        f'<div class="feat-chip"><span class="feat-name">{name}</span>'
        f'<span class="feat-val">{val}</span></div>'
        for name, val in all_features
    )

    shown_trees = min(6, n_estimators) if isinstance(n_estimators, int) else 6
    remaining_trees = (n_estimators - shown_trees) if isinstance(n_estimators, int) else None
    tree_icons_html = "".join(
        f'''<div class="tree-icon" id="tree-{i}">
            <svg viewBox="0 0 40 50" width="30" height="38">
                <polygon points="20,2 34,22 6,22" fill="#16a34a"/>
                <polygon points="20,14 32,32 8,32" fill="#22c55e"/>
                <rect x="17" y="32" width="6" height="14" fill="#78350f"/>
            </svg>
            <span>Tree {i+1}</span>
        </div>'''
        for i in range(shown_trees)
    )
    remaining_label = f"⋮ +{remaining_trees} more trees" if remaining_trees else ""

    def sx(z): return 10 + (z + 6) / 12 * 200
    def sy(s): return 130 - s * 120
    sigmoid_polyline = " ".join(
        f"{sx(-6 + (i/40)*12):.1f},{sy(1/(1+math.exp(-(-6 + (i/40)*12)))):.1f}"
        for i in range(41)
    )
    marker_z = max(-6, min(6, raw_score))
    marker_x, marker_y = sx(marker_z), sy(prob)

    class_color = "#dc2626" if pred == 1 else "#16a34a"
    class_bg = "#fee2e2" if pred == 1 else "#dcfce7"
    class_text = "Higher Risk — Class 1" if pred == 1 else "Lower Risk — Class 0"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            padding: 20px;
        }}
        .title {{
            font-size: 19px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 2px;
        }}
        .subtitle {{
            font-size: 12.5px;
            color: #64748b;
            margin-bottom: 12px;
        }}
        .controls {{
            margin-bottom: 16px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .sim-btn {{
            background: #0f172a;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .sim-btn:hover {{ background: #1e293b; }}
        .sim-btn:disabled {{ background: #94a3b8; cursor: not-allowed; }}
        .status {{
            font-size: 12.5px;
            color: #475569;
            font-family: 'Consolas', monospace;
        }}
        .pipeline {{
            display: flex;
            align-items: stretch;
            gap: 0;
            overflow-x: auto;
            padding-bottom: 8px;
        }}
        .layer-card {{
            min-width: 230px;
            max-width: 230px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.06);
            display: flex;
            flex-direction: column;
            height: 480px;
            transition: all 0.4s ease;
            opacity: 0.55;
            transform: scale(0.97);
        }}
        .layer-card.active {{
            opacity: 1;
            transform: scale(1);
            box-shadow: 0 0 0 3px rgba(59,130,246,0.35), 0 8px 20px rgba(15,23,42,0.12);
        }}
        .layer-card.done {{
            opacity: 1;
            transform: scale(1);
        }}
        .layer-header {{
            padding: 9px 12px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.4px;
            color: white;
            border-radius: 10px 10px 0 0;
        }}
        .layer-sub {{
            font-size: 10.5px;
            color: #94a3b8;
            padding: 6px 12px 0 12px;
        }}
        .layer-body {{
            padding: 10px 12px;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            overflow-y: auto;
        }}
        .layer-body.center {{
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .connector {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            padding: 0 4px;
            opacity: 0.4;
            transition: opacity 0.4s;
        }}
        .connector.active {{ opacity: 1; }}
        .connector-line {{
            width: 100%;
            height: 2px;
            background: repeating-linear-gradient(90deg, #94a3b8 0 6px, transparent 6px 11px);
            position: relative;
        }}
        .connector-line::after {{
            content: "";
            position: absolute;
            right: -2px;
            top: -4px;
            border: 5px solid transparent;
            border-left-color: #64748b;
        }}
        .connector-label {{
            font-size: 10px;
            color: #475569;
            text-align: center;
            margin-top: 6px;
            line-height: 1.35;
            font-family: 'Consolas', monospace;
        }}

        .feat-chip {{
            display: flex;
            justify-content: space-between;
            font-size: 10.5px;
            padding: 4px 0;
            border-bottom: 1px dashed #f1f5f9;
            opacity: 0;
            transform: translateX(-8px);
            transition: all 0.3s ease;
        }}
        .feat-chip.show {{
            opacity: 1;
            transform: translateX(0);
        }}
        .feat-name {{ color: #475569; }}
        .feat-val {{ color: #0f172a; font-weight: 600; font-family: 'Consolas', monospace; }}

        .tree-icon {{
            display: flex;
            flex-direction: column;
            align-items: center;
            font-size: 9.5px;
            color: #475569;
            display: inline-flex;
            width: 33%;
            margin-bottom: 6px;
            opacity: 0.25;
            transform: scale(0.8);
            transition: all 0.35s ease;
        }}
        .tree-icon.lit {{
            opacity: 1;
            transform: scale(1);
        }}
        .tree-grid {{ display: flex; flex-wrap: wrap; }}
        .tree-remaining {{
            font-size: 10.5px;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .hparams {{
            font-size: 10px;
            color: #64748b;
            border-top: 1px solid #f1f5f9;
            margin-top: 10px;
            padding-top: 8px;
            line-height: 1.6;
        }}

        .formula {{
            font-family: 'Consolas', monospace;
            font-size: 12.5px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 10px;
            margin-bottom: 10px;
        }}
        .big-value {{
            font-size: 22px;
            font-weight: 700;
            color: #ea580c;
            font-family: 'Consolas', monospace;
            opacity: 0;
            transition: opacity 0.5s;
        }}
        .big-value.show {{ opacity: 1; }}
        .value-label {{
            font-size: 10.5px;
            color: #64748b;
            margin-top: 4px;
        }}

        #sigmoid-marker {{
            opacity: 0;
            transition: opacity 0.6s;
        }}
        #sigmoid-marker.show {{ opacity: 1; }}

        .prob-bar-track {{
            width: 100%;
            height: 10px;
            background: #f1f5f9;
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0;
        }}
        .prob-bar-fill {{
            height: 100%;
            border-radius: 6px;
            width: 0%;
            transition: width 1.2s ease-out;
        }}
        .class-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 12.5px;
            margin-top: 10px;
            opacity: 0;
            transform: scale(0.8);
            transition: all 0.4s ease;
        }}
        .class-badge.show {{
            opacity: 1;
            transform: scale(1);
        }}
        .threshold-note {{
            font-size: 10px;
            color: #94a3b8;
            margin-top: 8px;
        }}
    </style>
    </head>
    <body>

    <div class="title">XGBoost Ensemble Architecture — Single-Patient Inference Trace</div>
    <div class="subtitle">Predicted Probability = {prob:.1%} &nbsp;|&nbsp; Predicted Class = {class_text}</div>

    <div class="controls">
        <button class="sim-btn" id="runBtn" onclick="runSimulation()">▶ Run Simulation</button>
        <span class="status" id="status">Ready</span>
    </div>

    <div class="pipeline">

        <div class="layer-card" id="layer-input">
            <div class="layer-header" style="background:#0284c7;">INPUT LAYER</div>
            <div class="layer-sub">Patient feature vector (14 features)</div>
            <div class="layer-body">
                {feature_grid_html}
            </div>
        </div>

        <div class="connector" id="conn-1">
            <div class="connector-line"></div>
            <div class="connector-label">x ∈ ℝ¹⁴<br>(scaled/encoded)</div>
        </div>

        <div class="layer-card" id="layer-ensemble">
            <div class="layer-header" style="background:#16a34a;">ENSEMBLE LAYER</div>
            <div class="layer-sub">Gradient-boosted decision trees</div>
            <div class="layer-body">
                <div class="tree-grid">
                    {tree_icons_html}
                </div>
                <div class="tree-remaining">{remaining_label}</div>
                <div class="hparams">
                    n_estimators = {n_estimators}<br>
                    max_depth = {max_depth}<br>
                    learning_rate = {learning_rate}<br>
                    Each tree sequentially corrects the residual error of the previous trees.
                </div>
            </div>
        </div>

        <div class="connector" id="conn-2">
            <div class="connector-line"></div>
            <div class="connector-label">K leaf<br>outputs f<sub>k</sub>(x)</div>
        </div>

        <div class="layer-card" id="layer-agg">
            <div class="layer-header" style="background:#ea580c;">AGGREGATION LAYER</div>
            <div class="layer-sub">Sum of leaf scores → raw score</div>
            <div class="layer-body center">
                <div class="formula">F(x) = Σ<sub>k=1</sub><sup>K</sup> f<sub>k</sub>(x)</div>
                <div class="big-value" id="raw-score">z = {raw_score:.3f}</div>
                <div class="value-label">Raw score (logit), recovered<br>exactly from the model's output</div>
            </div>
        </div>

        <div class="connector" id="conn-3">
            <div class="connector-line"></div>
            <div class="connector-label">raw score<br>z = {raw_score:.3f}</div>
        </div>

        <div class="layer-card" id="layer-act">
            <div class="layer-header" style="background:#a855f7;">ACTIVATION LAYER</div>
            <div class="layer-sub">Sigmoid function</div>
            <div class="layer-body center">
                <div class="formula">σ(z) = 1 / (1 + e<sup>−z</sup>)</div>
                <svg viewBox="0 0 220 140" width="200" height="126">
                    <line x1="10" y1="130" x2="210" y2="130" stroke="#cbd5e1" stroke-width="1"/>
                    <line x1="110" y1="10" x2="110" y2="130" stroke="#cbd5e1" stroke-width="1"/>
                    <polyline points="{sigmoid_polyline}" fill="none" stroke="#a855f7" stroke-width="2.5"/>
                    <g id="sigmoid-marker">
                        <line x1="{marker_x:.1f}" y1="130" x2="{marker_x:.1f}" y2="{marker_y:.1f}" stroke="#f472b6" stroke-width="1" stroke-dasharray="3,3"/>
                        <line x1="10" y1="{marker_y:.1f}" x2="{marker_x:.1f}" y2="{marker_y:.1f}" stroke="#f472b6" stroke-width="1" stroke-dasharray="3,3"/>
                        <circle cx="{marker_x:.1f}" cy="{marker_y:.1f}" r="4.5" fill="#f472b6"/>
                    </g>
                </svg>
                <div class="value-label">This patient's exact position<br>on the sigmoid curve</div>
            </div>
        </div>

        <div class="connector" id="conn-4">
            <div class="connector-line"></div>
            <div class="connector-label">p = σ(z)<br>= {prob:.1%}</div>
        </div>

        <div class="layer-card" id="layer-out">
            <div class="layer-header" style="background:{class_color};">OUTPUT LAYER</div>
            <div class="layer-sub">Predicted probability &amp; class</div>
            <div class="layer-body center">
                <div class="big-value" id="final-prob" style="color:{class_color};">{prob:.1%}</div>
                <div class="value-label">Predicted probability of diabetes</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="prob-bar" style="background:{class_color};"></div>
                </div>
                <div class="class-badge" id="class-badge" style="background:{class_bg}; color:{class_color};">{class_text}</div>
                <div class="threshold-note">Decision threshold = 0.5<br>p {'≥' if pred == 1 else '<'} 0.5 → Class {pred}</div>
            </div>
        </div>

    </div>

    <script>
    function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

    async function runSimulation() {{
        const btn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        btn.disabled = true;
        status.textContent = "Running inference simulation...";

        // Reset
        document.querySelectorAll('.layer-card').forEach(el => el.classList.remove('active', 'done'));
        document.querySelectorAll('.connector').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.feat-chip').forEach(el => el.classList.remove('show'));
        document.querySelectorAll('.tree-icon').forEach(el => el.classList.remove('lit'));
        document.getElementById('raw-score').classList.remove('show');
        document.getElementById('sigmoid-marker').classList.remove('show');
        document.getElementById('prob-bar').style.width = '0%';
        document.getElementById('class-badge').classList.remove('show');
        document.getElementById('final-prob').classList.remove('show');

        // Step 1
        status.textContent = "1/5  Input features loading...";
        const inputLayer = document.getElementById('layer-input');
        inputLayer.classList.add('active');
        const chips = document.querySelectorAll('.feat-chip');
        for (let i = 0; i < chips.length; i++) {{
            chips[i].classList.add('show');
            await sleep(60);
        }}
        await sleep(600);
        inputLayer.classList.remove('active');
        inputLayer.classList.add('done');
        document.getElementById('conn-1').classList.add('active');

        // Step 2
        status.textContent = "2/5  Trees evaluating features...";
        const ensLayer = document.getElementById('layer-ensemble');
        ensLayer.classList.add('active');
        const trees = document.querySelectorAll('.tree-icon');
        for (let i = 0; i < trees.length; i++) {{
            trees[i].classList.add('lit');
            await sleep(220);
        }}
        await sleep(700);
        ensLayer.classList.remove('active');
        ensLayer.classList.add('done');
        document.getElementById('conn-2').classList.add('active');

        // Step 3
        status.textContent = "3/5  Aggregating leaf scores → raw score...";
        const aggLayer = document.getElementById('layer-agg');
        aggLayer.classList.add('active');
        document.getElementById('raw-score').classList.add('show');
        await sleep(1100);
        aggLayer.classList.remove('active');
        aggLayer.classList.add('done');
        document.getElementById('conn-3').classList.add('active');

        // Step 4
        status.textContent = "4/5  Applying sigmoid activation...";
        const actLayer = document.getElementById('layer-act');
        actLayer.classList.add('active');
        document.getElementById('sigmoid-marker').classList.add('show');
        await sleep(1200);
        actLayer.classList.remove('active');
        actLayer.classList.add('done');
        document.getElementById('conn-4').classList.add('active');

        // Step 5
        status.textContent = "5/5  Final prediction ready";
        const outLayer = document.getElementById('layer-out');
        outLayer.classList.add('active');
        document.getElementById('final-prob').classList.add('show');
        document.getElementById('prob-bar').style.width = '{prob*100:.1f}%';
        await sleep(900);
        document.getElementById('class-badge').classList.add('show');
        await sleep(800);

        status.textContent = "Simulation complete ✓";
        btn.disabled = false;
    }}
    </script>
    </body>
    </html>
    """

    components.html(html_code, height=640, scrolling=True)