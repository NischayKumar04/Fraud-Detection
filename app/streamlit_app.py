import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

from src.utils import load_joblib, MODELS_DIR
from src.predict import predict_batch
from src.data_loader import load_clean_train
from src.features import split_xy

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("💳 Fraud Detection Demo")

# ---------- Load artifacts ----------
info_path = MODELS_DIR / "best_model_info.json"
metrics_path = MODELS_DIR / "metrics.json"
model_path = MODELS_DIR / "best_model.joblib"

info = json.loads(Path(info_path).read_text()) if info_path.exists() else {}
metrics = json.loads(Path(metrics_path).read_text()) if metrics_path.exists() else {}
model = load_joblib(model_path) if model_path.exists() else None

best_model = info.get("best_model", "N/A")
threshold = float(info.get("best_threshold", 0.5))

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model Info")
    st.metric("Best Model", best_model)
    st.metric("Threshold", f"{threshold:.3f}")
    st.caption(f"Mode: {info.get('threshold_mode', 'N/A')}, FN:FP = {info.get('fn_cost','?')}:{info.get('fp_cost','?')}")

    # Try to show headline metrics for best model
    m = metrics.get("models", {}).get(best_model, {})
    sel = m.get("selected_metrics", {})
    if sel:
        st.metric("PR-AUC", f"{sel.get('pr_auc', np.nan):.3f}")
        st.metric("Recall (Class 1)", f"{sel.get('recall', np.nan):.3f}")
        st.metric("Precision (Class 1)", f"{sel.get('precision', np.nan):.3f}")

st.caption(f"Decision threshold: **{threshold:.3f}** (loaded from `best_model_info.json`)")

# ---------- Prepare defaults ----------
@st.cache_data
def get_feature_defaults():
    df = load_clean_train("data/clean_train.csv")
    X, y = split_xy(df, target_col="isFraud")
    med = X.median(numeric_only=True)
    modes = {c: X[c].mode(dropna=True).iloc[0] for c in X.columns if c not in med.index and not X[c].mode(dropna=True).empty}
    return X.columns.tolist(), med.to_dict(), modes, X, y

cols, medians, modes, X_all, y_all = get_feature_defaults()

def base_input():
    row = {}
    for c in cols:
        if c in medians:
            row[c] = float(medians[c])
        else:
            row[c] = modes.get(c, 0)
    return row

# sample transactions (edit keys based on your exact features)
def legit_sample():
    r = base_input()
    for k, v in {
        "TransactionAmt": 38.0,
        "hour": 13,
        "is_high_amt": 0,
        "card4_freq": 0.2,
        "P_emaildomain_freq": 0.3,
    }.items():
        if k in r:
            r[k] = v
    return r

def fraud_sample():
    r = base_input()
    for k, v in {
        "TransactionAmt": 1499.0,
        "hour": 2,
        "is_high_amt": 1,
        "card4_freq": 0.01,
        "P_emaildomain_freq": 0.01,
    }.items():
        if k in r:
            r[k] = v
    return r

if "input_row" not in st.session_state:
    st.session_state.input_row = base_input()

c1, c2 = st.columns(2)
if c1.button("Load Legitimate Sample"):
    st.session_state.input_row = legit_sample()
if c2.button("Load Fraud Sample"):
    st.session_state.input_row = fraud_sample()

# ---------- Minimal editable form ----------
st.subheader("Transaction Input")
edit_keys = [k for k in ["TransactionAmt", "hour", "is_high_amt", "card4_freq", "P_emaildomain_freq"] if k in cols]
for k in edit_keys:
    val = st.session_state.input_row[k]
    if isinstance(val, (int, np.integer)):
        st.session_state.input_row[k] = st.number_input(k, value=int(val), step=1)
    else:
        st.session_state.input_row[k] = st.number_input(k, value=float(val))

input_df = pd.DataFrame([st.session_state.input_row], columns=cols)

# ---------- Predict ----------
if st.button("Predict Fraud Risk", type="primary"):
    pred = predict_batch(input_df)
    fraud_prob = float(pred["fraud_probability"].iloc[0])
    fraud_pred = int(pred["fraud_prediction"].iloc[0])

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_prob * 100,
        number={"suffix": "%"},
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if fraud_prob >= threshold else "green"},
            "steps": [
                {"range": [0, threshold * 100], "color": "lightgreen"},
                {"range": [threshold * 100, 100], "color": "lightsalmon"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**Fraud probability:** {fraud_prob:.4f}")
    st.write(f"**Prediction:** {'🚨 Fraud' if fraud_pred == 1 else '✅ Legitimate'}")

    # Optional SHAP explanation
    with st.expander("Show SHAP explanation for this prediction"):
        if model is None:
            st.warning("Model not loaded.")
        else:
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(input_df)

                # handle class outputs
                if isinstance(sv, list):
                    sv1 = sv[1] if len(sv) > 1 else sv[0]
                    exp = shap.Explanation(
                        values=sv1[0],
                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=input_df.columns.tolist(),
                    )
                else:
                    vals = sv.values if hasattr(sv, "values") else sv
                    if vals.ndim == 3:
                        vals = vals[:, :, 1]
                    exp = shap.Explanation(
                        values=vals[0],
                        base_values=sv.base_values[0] if hasattr(sv, "base_values") else 0.0,
                        data=input_df.iloc[0].values,
                        feature_names=input_df.columns.tolist(),
                    )

                plt.figure(figsize=(10, 5))
                shap.plots.waterfall(exp, show=False, max_display=12)
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.warning(f"SHAP plot unavailable for this model/output format: {e}")