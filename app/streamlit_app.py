import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODELS_DIR = Path("models")
DATA_PATH = Path("data/clean_train.csv")

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("🔍 Fraud Detection Demo")

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    threshold = 0.5
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            threshold = float(info.get("best_threshold", 0.5))
    return model, threshold

@st.cache_data
def load_sample_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, nrows=2000)
        return df
    return None

model, saved_threshold = load_model()
df_sample = load_sample_data()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.01, 0.99, float(saved_threshold), 0.01)

st.subheader("Option A: Predict from uploaded CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    in_df = pd.read_csv(uploaded)
    X = in_df.drop(columns=["isFraud", "TransactionID"], errors="ignore").copy()
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = in_df.copy()
    out["fraud_probability"] = probs
    out["fraud_prediction"] = preds

    st.success(f"Scored {len(out)} rows")
    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
    )

st.markdown("---")
st.subheader("Option B: Demo on sample rows")

if df_sample is not None:
    demo_n = st.slider("Rows to score from sample", 5, 200, 20)
    demo_df = df_sample.head(demo_n).copy()
    X_demo = demo_df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    probs = model.predict_proba(X_demo)[:, 1]
    preds = (probs >= threshold).astype(int)

    demo_df["fraud_probability"] = probs
    demo_df["fraud_prediction"] = preds

    st.dataframe(demo_df, use_container_width=True)
else:
    st.info("No data/clean_train.csv found. Upload a CSV above.")