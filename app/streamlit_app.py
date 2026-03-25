import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# ---- Make repo root importable ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_joblib, MODELS_DIR
from src.data_loader import load_clean_train
from src.features import split_xy


# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.caption("Cost-sensitive operating point: FN=25, FP=1")

INFO_PATH = MODELS_DIR / "best_model_info.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

FN_COST = 25
FP_COST = 1


# =========================================================
# Cached loaders
# =========================================================
@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_joblib(MODEL_PATH)


@st.cache_data
def get_info_metrics():
    info = {}
    metrics = {}
    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text())
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    return info, metrics


@st.cache_data
def get_eval_pool():
    """Time-based holdout pool (last 20%)."""
    df = load_clean_train("data/clean_train.csv")
    X, y = split_xy(df, target_col="isFraud")
    n = len(X)
    cut = int(0.8 * n)
    return X.iloc[cut:].copy(), y.iloc[cut:].copy()


def predict_proba_df(model, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_df)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    s = model.decision_function(X_df)
    s = np.asarray(s)
    return 1.0 / (1.0 + np.exp(-s))


@st.cache_data
def get_ranked_samples():
    """
    Fraud samples: highest-risk fraud rows first.
    Legit samples: lowest-risk legit rows first.
    """
    model = get_model()
    X_te, y_te = get_eval_pool()

    probs = predict_proba_df(model, X_te)
    tmp = X_te.copy()
    tmp["_prob"] = probs
    tmp["_y"] = y_te.values

    fraud_pool = tmp[tmp["_y"] == 1].sort_values("_prob", ascending=False).drop(columns=["_prob", "_y"])
    legit_pool = tmp[tmp["_y"] == 0].sort_values("_prob", ascending=True).drop(columns=["_prob", "_y"])
    return fraud_pool, legit_pool


@st.cache_resource
def get_shap_explainer():
    """
    Cached SHAP explainer with small background for speed.
    """
    model = get_model()
    X_te, _ = get_eval_pool()
    bg = X_te.sample(n=min(200, len(X_te)), random_state=42)  # small background = faster
    try:
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    except Exception:
        explainer = shap.Explainer(model, bg)
    return explainer


# =========================================================
# Load artifacts
# =========================================================
model = get_model()
if model is None:
    st.error("Model not found at models/best_model.joblib. Please train first.")
    st.stop()

info, metrics = get_info_metrics()
best_model_name = info.get("best_model", "N/A")
threshold = float(info.get("best_threshold", 0.293))  # fallback for fn=25 run
threshold_mode = info.get("threshold_mode", "cost")

# Sidebar
with st.sidebar:
    st.header("Model Info")
    st.metric("Best Model", str(best_model_name))
    st.metric("Threshold", f"{threshold:.3f}")
    st.caption(f"Threshold mode: {threshold_mode}")
    st.caption(f"Cost policy shown: FN={FN_COST}, FP={FP_COST}")

    mm = metrics.get("models", {}).get(best_model_name, {})
    sm = mm.get("selected_metrics", {})
    if sm:
        st.metric("PR-AUC", f"{sm.get('pr_auc', np.nan):.3f}")
        st.metric("ROC-AUC", f"{sm.get('roc_auc', np.nan):.3f}")
        st.metric("Recall (Class 1)", f"{sm.get('recall', np.nan):.3f}")
        st.metric("Precision (Class 1)", f"{sm.get('precision', np.nan):.3f}")


# =========================================================
# Session state
# =========================================================
fraud_pool, legit_pool = get_ranked_samples()

if "input_row" not in st.session_state:
    st.session_state.input_row = legit_pool.iloc[0].to_dict()

if "fraud_click_i" not in st.session_state:
    st.session_state.fraud_click_i = 0
if "legit_click_i" not in st.session_state:
    st.session_state.legit_click_i = 0


# =========================================================
# Sample buttons
# =========================================================
c1, c2 = st.columns(2)

if c1.button("Load Legitimate Sample"):
    i = st.session_state.legit_click_i % max(len(legit_pool), 1)
    st.session_state.input_row = legit_pool.iloc[i].to_dict()
    st.session_state.legit_click_i += 1
    st.rerun()

if c2.button("Load Fraud Sample"):
    i = st.session_state.fraud_click_i % max(len(fraud_pool), 1)
    st.session_state.input_row = fraud_pool.iloc[i].to_dict()
    st.session_state.fraud_click_i += 1
    st.rerun()

st.caption(
    f"Legit sample index: {st.session_state.legit_click_i} | "
    f"Fraud sample index: {st.session_state.fraud_click_i}"
)


# =========================================================
# Input editor (key fields only)
# =========================================================
st.subheader("Transaction Input")
all_cols = list(st.session_state.input_row.keys())
key_fields = [k for k in ["TransactionAmt", "hour", "is_high_amt", "card4_freq", "P_emaildomain_freq"] if k in all_cols]

if key_fields:
    ui_cols = st.columns(min(3, len(key_fields)))
    for idx, k in enumerate(key_fields):
        with ui_cols[idx % len(ui_cols)]:
            v = st.session_state.input_row[k]
            if isinstance(v, (int, np.integer)):
                st.session_state.input_row[k] = st.number_input(k, value=int(v), step=1)
            else:
                st.session_state.input_row[k] = st.number_input(k, value=float(v))
else:
    st.info("Key engineered fields not found in current features; using loaded sample values.")

input_df = pd.DataFrame([st.session_state.input_row], columns=all_cols)


# =========================================================
# Predict + Gauge + SHAP
# =========================================================
if st.button("Predict Fraud Risk", type="primary"):
    prob = float(predict_proba_df(model, input_df)[0])
    pred = int(prob >= threshold)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if pred == 1 else "green"},
            "steps": [
                {"range": [0, threshold * 100], "color": "lightgreen"},
                {"range": [threshold * 100, 100], "color": "lightsalmon"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
        },
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.write(f"**Fraud probability:** `{prob:.4f}`")
    st.write(f"**Decision threshold:** `{threshold:.3f}` (cost-mode FN={FN_COST}, FP={FP_COST})")
    st.write(f"**Prediction:** {'🚨 Fraud' if pred == 1 else '✅ Legitimate'}")

    # Keep SHAP, but gated to reduce lag
    if st.checkbox("Show SHAP explanation (slower)", value=False):
        try:
            explainer = get_shap_explainer()
            sv = explainer(input_df)

            vals = sv.values if hasattr(sv, "values") else np.asarray(sv)
            base_vals = sv.base_values if hasattr(sv, "base_values") else 0.0

            # Convert to class-1 vector if multiclass-like shape
            if isinstance(vals, list):
                vals = np.asarray(vals[1] if len(vals) > 1 else vals[0])

            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[:, :, 1]  # class 1
            one_vals = vals[0] if vals.ndim == 2 else vals

            if isinstance(base_vals, (list, np.ndarray)):
                base_val = np.asarray(base_vals).flatten()[0]
            else:
                base_val = float(base_vals)

            exp = shap.Explanation(
                values=one_vals,
                base_values=base_val,
                data=input_df.iloc[0].values,
                feature_names=input_df.columns.tolist(),
            )

            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(exp, max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")