import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Engine Failure Prediction", layout="centered")
st.title("Engine Failure Prediction 🚀")

from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="chaitram/Engine-Failure-Prediction",
    filename="best_engine_failure_model_v1.joblib",
    repo_type="model"
)

# If your dataset is public, HF filesystem URL can work inside Spaces
DATASET_PATH = "hf://datasets/chaitram/Engine-Failure-Prediction/engine_data.csv"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_dataset():
    """
    Used only to populate dropdown choices.
    If it fails (no HF filesystem / dependency), we fall back to manual choices.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception:
        return None

def build_feature_vector(raw_df: pd.DataFrame, model):
    """
    Makes the input match what the saved model expects.
    Case A: model is a Pipeline that expects raw columns -> use raw_df directly if it matches.
    Case B: model is trained on get_dummies output -> one-hot encode raw_df and align to feature_names_in_.
    """
    required = getattr(model, "feature_names_in_", None)

    # If model doesn't expose required columns, just return raw_df
    if required is None:
        return raw_df

    required = list(required)

    # If required columns look like raw columns and are all present -> reorder and return
    if all(col in raw_df.columns for col in required):
        return raw_df[required]

    # Otherwise assume model expects one-hot encoded columns
    encoded = pd.get_dummies(raw_df, drop_first=True)

    # Add any missing columns (model expects them) as 0
    for col in required:
        if col not in encoded.columns:
            encoded[col] = 0

    # Drop extra columns and order correctly
    encoded = encoded[required]
    return encoded

def safe_predict(model, X):
    """
    Returns prediction + probability (if classifier supports predict_proba).
    """
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
    return pred, proba

# -----------------------------
# Load model + reference
# -----------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load {MODEL_PATH}. Make sure best_engine_failure_model_v1.joblib is uploaded to the Space.\n\nError: {e}")
    st.stop()

ref = load_reference_dataset()

# Build dropdown options from dataset if available; otherwise provide simple fallbacks
def get_options(col_name, fallback):
    if ref is not None and col_name in ref.columns:
        vals = sorted([v for v in ref[col_name].dropna().unique().tolist()])
        return vals if len(vals) > 0 else fallback
    return fallback

# -----------------------------
# UI Inputs (NO SPACES IN VARIABLE NAMES)
# -----------------------------
st.subheader("Enter Engine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM (Revolutions Per Minute)", min_value=0, max_value=10000, value=1500, step=10)
    lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar/kPa)", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
    fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

with col2:
    coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    lub_oil_temperature = st.number_input("Lubricating Oil Temperature (°C)", min_value=0.0, max_value=200.0, value=85.0, step=0.5)
    coolant_temperature = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=90.0, step=0.5)

# -----------------------------
# Assemble input into DataFrame (FIXED)
# -----------------------------
raw_input_df = pd.DataFrame([{
    "Engine_RPM":           int(engine_rpm),
    "Lub_Oil_Pressure":     float(lub_oil_pressure),
    "Fuel_Pressure":        float(fuel_pressure),
    "Coolant_Pressure":     float(coolant_pressure),
    "Lub_Oil_Temperature":  float(lub_oil_temperature),
    "Coolant_Temperature":  float(coolant_temperature),
}])

with st.expander("Show input data"):
    st.dataframe(raw_input_df, use_container_width=True)


# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Engine Condition"):
    try:
        X = build_feature_vector(raw_input_df, model)
        pred, proba = safe_predict(model, X)

        result = "Maintenance Required" if int(pred) == 1 else "Engine Operating Normally"

        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(
            "Prediction failed. This usually happens when the model expects different feature columns "
            "(e.g., one-hot columns) than the app is sending.\n\n"
            f"Error: {e}"
        )
