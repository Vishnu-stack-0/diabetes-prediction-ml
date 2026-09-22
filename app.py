import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "diabetes_pipeline.joblib"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()
preproc = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

st.title("Diabetes Risk Prediction")
st.caption("Enter patient details to estimate diabetes risk. Hover the ⓘ on any field for the normal range.")

with st.expander("📋 What do these numbers mean? "):
    st.markdown("""
| Field | Normal range (roughly) | Notes |
|---|---|---|
| **Age** | any adult age | in years |
| **Pulse rate** | 60–100 bpm | resting heart rate |
| **Systolic BP** | 90–120 mmHg | "top" blood pressure number; >140 is high |
| **Diastolic BP** | 60–80 mmHg | "bottom" blood pressure number; >90 is high |
| **Glucose** | 4–6 mmol/L (fasting) | fasting blood sugar; ≥7 mmol/L is typically diabetic range |
| **Height** | — | in metres, e.g. 1.65 for 165 cm |
| **Weight** | — | in kilograms |
| **BMI** | 18.5–24.9 = normal, 25–29.9 = overweight, 30+ = obese | calculated from height & weight |
| **Family diabetes / hypertensive / family hypertension / cardiovascular disease / stroke** | 0 = No, 1 = Yes | tick based on medical history |

These ranges are general reference points, not a diagnosis on their own.
""")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=45,
            help="Patient's age in years."
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        pulse_rate = st.number_input(
            "Pulse rate (bpm)", min_value=30, max_value=200, value=76,
            help="Resting heart rate. Normal range: 60–100 bpm."
        )
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)", min_value=70, max_value=220, value=130,
            help="The higher number in a blood pressure reading (e.g. the '120' in 120/80). Normal: 90–120 mmHg."
        )
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)", min_value=40, max_value=130, value=81,
            help="The lower number in a blood pressure reading (e.g. the '80' in 120/80). Normal: 60–80 mmHg."
        )
        glucose = st.number_input(
            "Fasting glucose (mmol/L)", min_value=0.0, max_value=40.0, value=6.9, step=0.1,
            help="Blood sugar level after fasting. Normal: 4–6 mmol/L. 6–7 = pre-diabetic range. 7+ = diabetic range."
        )
        height = st.number_input(
            "Height (metres)", min_value=1.0, max_value=2.2, value=1.55, step=0.01,
            help="Enter height in metres — e.g. for 165 cm, type 1.65."
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=200.0, value=58.0, step=0.5,
            help="Body weight in kilograms."
        )
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1,
            help="Body Mass Index. Under 18.5 = underweight, 18.5–24.9 = normal, 25–29.9 = overweight, 30+ = obese."
        )
        family_diabetes = st.selectbox(
            "Family history of diabetes", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has a close family member (parent/sibling) been diagnosed with diabetes?"
        )
        hypertensive = st.selectbox(
            "Currently hypertensive", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has the patient been diagnosed with high blood pressure?"
        )
        family_hypertension = st.selectbox(
            "Family history of hypertension", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has a close family member been diagnosed with high blood pressure?"
        )
        cardiovascular_disease = st.selectbox(
            "Cardiovascular disease", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has the patient been diagnosed with any heart/blood vessel disease?"
        )
        stroke = st.selectbox(
            "History of stroke", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has the patient had a stroke previously?"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "pulse_rate": pulse_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "family_diabetes": family_diabetes,
        "hypertensive": hypertensive,
        "family_hypertension": family_hypertension,
        "cardiovascular_disease": cardiovascular_disease,
        "stroke": stroke
    }])

    prob = pipeline.predict_proba(input_df)[0, 1]
    pred = pipeline.predict(input_df)[0]

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.error(f"Higher diabetes risk — estimated probability: {prob:.1%}")
    else:
        st.success(f"Lower diabetes risk — estimated probability: {prob:.1%}")

    st.progress(min(int(prob * 100), 100))

    # Plain-language flags for anything outside the normal range entered above
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

    st.divider()
    st.subheader("Why this prediction?")
    st.caption("Each bar shows how much that specific value pushed the prediction up (red, toward higher risk) or down (blue, toward lower risk).")

    input_transformed = preproc.transform(input_df)
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()
    feature_names = preproc.get_feature_names_out()

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    shap_values = explainer.shap_values(input_transformed)

    base_value = (
        explainer.expected_value
        if np.isscalar(explainer.expected_value)
        else explainer.expected_value[1]
    )
    sample_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=input_transformed[0],
        feature_names=feature_names,
    )

    shap.plots.waterfall(sample_explanation, show=False)
    st.pyplot(plt.gcf())
    
    