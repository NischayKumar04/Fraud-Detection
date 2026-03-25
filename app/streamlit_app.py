import sys
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---- Make repo root importable ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_joblib, MODELS_DIR
from src.data_loader import load_clean_train
from src.features import split_xy


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.caption("Cost-sensitive operating point: FN=25, FP=1")


# =========================================================
# Paths / constants
# =========================================================
INFO_PATH    = MODELS_DIR / "best_model_info.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH   = MODELS_DIR / "best_model.joblib"

FN_COST = 25
FP_COST = 1

KEY_FIELDS_CANDIDATES = [
    "TransactionAmt", "hour", "is_high_amt", "card4_freq", "P_emaildomain_freq"
]


# =========================================================
# Utility: robust numeric coercion
# =========================================================
_num_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_numeric_like(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return np.nan
    s_clean = s.replace("[", " ").replace("]", " ").replace(",", " ").strip()
    m = _num_pattern.search(s_clean)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return np.nan
    return np.nan


def coerce_df_numeric(df: pd.DataFrame, fallback_medians: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(parse_numeric_like)
    out = out.astype("float32")
    out = out.fillna(fallback_medians)
    return out


# =========================================================
# Cached helpers
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
    df = load_clean_train("data/clean_train.csv")
    X, y = split_xy(df, target_col="isFraud")
    cut  = int(0.8 * len(X))
    return X.iloc[cut:].copy(), y.iloc[cut:].copy()


def predict_proba_df(model, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_df)
        return p[:, 1] if (p.ndim == 2 and p.shape[1] >= 2) else p.ravel()
    s = np.asarray(model.decision_function(X_df))
    return 1.0 / (1.0 + np.exp(-s))


@st.cache_data
def get_ranked_samples_and_probs():
    """
    Coerces the eval pool at source so no downstream code ever sees
    object-dtype strings like '[5E-1]'.
    Returns: fraud_pool, legit_pool, all_probs, all_labels, X_clean
    """
    model       = get_model()
    X_te, y_te  = get_eval_pool()
    feature_cols = X_te.columns.tolist()

    medians_tmp = (
        X_te.apply(lambda col: col.map(parse_numeric_like))
        .median()
        .reindex(feature_cols)
        .fillna(0.0)
        .astype("float32")
    )
    X_clean = coerce_df_numeric(X_te, medians_tmp)

    probs = predict_proba_df(model, X_clean)
    tmp   = X_clean.copy()
    tmp["_prob"] = probs
    tmp["_y"]    = y_te.values

    fraud_pool = (
        tmp[tmp["_y"] == 1]
        .sort_values("_prob", ascending=False)
        .drop(columns=["_prob", "_y"])
    )
    legit_pool = (
        tmp[tmp["_y"] == 0]
        .sort_values("_prob", ascending=True)
        .drop(columns=["_prob", "_y"])
    )
    return fraud_pool, legit_pool, probs, y_te.values, X_clean


def render_gauge(prob: float, threshold: float):
    pred = int(prob >= threshold)
    fig  = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "crimson" if pred == 1 else "seagreen"},
            "steps": [
                {"range": [0, threshold * 100],  "color": "lightgreen"},
                {"range": [threshold * 100, 100], "color": "lightsalmon"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Load model / artifacts
# =========================================================
model = get_model()
if model is None:
    st.error("Model not found at models/best_model.joblib — train first.")
    st.stop()

info, metrics   = get_info_metrics()
best_model_name = info.get("best_model", "N/A")
threshold       = float(info.get("best_threshold", 0.293))
threshold_mode  = info.get("threshold_mode", "cost")

X_te_ref, _     = get_eval_pool()
feature_order   = X_te_ref.columns.tolist()
feature_medians = (
    X_te_ref.apply(lambda col: col.map(parse_numeric_like))
    .median()
    .reindex(feature_order)
    .fillna(0.0)
    .astype("float32")
)

key_fields = [k for k in KEY_FIELDS_CANDIDATES if k in feature_order]

fraud_pool, legit_pool, all_probs, all_labels, X_clean = get_ranked_samples_and_probs()


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Model Info")
    st.metric("Best Model", str(best_model_name))
    st.metric("Threshold",  f"{threshold:.3f}")
    st.caption(f"Threshold mode: {threshold_mode}")
    st.caption(f"Cost policy: FN={FN_COST}, FP={FP_COST}")

    mm = metrics.get("models", {}).get(best_model_name, {})
    sm = mm.get("selected_metrics", {})
    if sm:
        st.metric("PR-AUC",              f"{sm.get('pr_auc',    np.nan):.3f}")
        st.metric("ROC-AUC",             f"{sm.get('roc_auc',   np.nan):.3f}")
        st.metric("Recall (Class 1)",    f"{sm.get('recall',    np.nan):.3f}")
        st.metric("Precision (Class 1)", f"{sm.get('precision', np.nan):.3f}")

st.caption(f"Threshold from `best_model_info.json`: **{threshold:.3f}**")


# =========================================================
# Session state init
# =========================================================
if "input_row"    not in st.session_state:
    st.session_state.input_row = legit_pool.iloc[0].reindex(feature_order).to_dict()
if "fraud_click_i"  not in st.session_state:
    st.session_state.fraud_click_i = 0
if "legit_click_i"  not in st.session_state:
    st.session_state.legit_click_i = 0
if "last_input_df"  not in st.session_state:
    st.session_state.last_input_df = None
if "last_prob"      not in st.session_state:
    st.session_state.last_prob = None
if "last_pred"      not in st.session_state:
    st.session_state.last_pred = None
if "sample_label"   not in st.session_state:
    st.session_state.sample_label = None   # "fraud" | "legit" | None


# =========================================================
# Helper: inject row into session state AND widget keys
# =========================================================
def _load_row(new_row: dict, label: str):
    st.session_state.input_row    = new_row
    st.session_state.sample_label = label
    # Write directly into each widget's session_state key.
    # st.number_input owns its value via key= in session_state; without this
    # injection the widget ignores input_row and keeps showing the old value.
    for k in key_fields:
        val = parse_numeric_like(new_row.get(k, feature_medians.get(k, 0.0)))
        if np.isnan(val):
            val = float(feature_medians.get(k, 0.0))
        st.session_state[f"inp_{k}"] = float(val)
    st.session_state.last_input_df = None
    st.session_state.last_prob     = None
    st.session_state.last_pred     = None


# =========================================================
# Sample buttons
# =========================================================
b1, b2 = st.columns(2)

if b1.button("✅ Load Legitimate Sample", use_container_width=True):
    i = st.session_state.legit_click_i % max(len(legit_pool), 1)
    _load_row(legit_pool.iloc[i].reindex(feature_order).to_dict(), "legit")
    st.session_state.legit_click_i += 1

if b2.button("🚨 Load Fraud Sample", use_container_width=True):
    i = st.session_state.fraud_click_i % max(len(fraud_pool), 1)
    _load_row(fraud_pool.iloc[i].reindex(feature_order).to_dict(), "fraud")
    st.session_state.fraud_click_i += 1

st.caption(
    f"Legit samples loaded: {st.session_state.legit_click_i} | "
    f"Fraud samples loaded: {st.session_state.fraud_click_i}"
)

if st.session_state.sample_label == "fraud":
    st.info("🚨 Currently showing a **fraud** sample from the holdout set.")
elif st.session_state.sample_label == "legit":
    st.info("✅ Currently showing a **legitimate** sample from the holdout set.")


# =========================================================
# Input editor
# =========================================================
st.subheader("Transaction Input")

if key_fields:
    cols_ui = st.columns(min(3, len(key_fields)))
    for idx, k in enumerate(key_fields):
        with cols_ui[idx % len(cols_ui)]:
            current = parse_numeric_like(
                st.session_state.input_row.get(k, feature_medians.get(k, 0.0))
            )
            if np.isnan(current):
                current = float(feature_medians.get(k, 0.0))
            st.session_state.input_row[k] = st.number_input(
                k, value=float(current), key=f"inp_{k}"
            )
else:
    st.info("Key engineered fields not found in feature set.")

raw_input_df   = pd.DataFrame([st.session_state.input_row]).reindex(columns=feature_order)
model_input_df = coerce_df_numeric(raw_input_df, feature_medians)


# =========================================================
# Predict
# =========================================================
if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
    prob = float(predict_proba_df(model, model_input_df)[0])
    pred = int(prob >= threshold)
    st.session_state.last_input_df = model_input_df.copy()
    st.session_state.last_prob     = prob
    st.session_state.last_pred     = pred


# =========================================================
# Result
# =========================================================
if st.session_state.last_input_df is not None:
    prob = st.session_state.last_prob
    pred = st.session_state.last_pred

    render_gauge(prob, threshold)

    r1, r2, r3 = st.columns(3)
    r1.metric("Fraud Probability", f"{prob:.4f}")
    r2.metric("Decision Threshold", f"{threshold:.3f}")
    r3.metric("Prediction", "🚨 Fraud" if pred == 1 else "✅ Legitimate")
else:
    st.info("Load a sample or edit inputs above, then click **Predict Fraud Risk**.")


# =========================================================
# Analytics
# =========================================================
st.divider()
st.header("📊 Model Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Score Distribution",
    "Transaction Amount",
    "Threshold vs Cost",
    "Confusion Matrix",
])

# ----------------------------------------------------------
# Tab 1 — Score Distribution
# ----------------------------------------------------------
with tab1:
    st.subheader("Fraud Probability Score Distribution")
    st.caption("How confidently the model separates fraud from legitimate on the holdout set.")

    fraud_scores = all_probs[all_labels == 1]
    legit_scores = all_probs[all_labels == 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=legit_scores, name="Legitimate",
        nbinsx=60, opacity=0.65,
        marker_color="seagreen",
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=fraud_scores, name="Fraud",
        nbinsx=60, opacity=0.75,
        marker_color="crimson",
        histnorm="probability density",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="black", line_width=2,
        annotation_text=f"Threshold {threshold:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Fraud Probability",
        yaxis_title="Density",
        legend=dict(orientation="h", y=1.1),
        height=380,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)
    ca.metric("Median score — Fraud",      f"{np.median(fraud_scores):.3f}")
    cb.metric("Median score — Legitimate", f"{np.median(legit_scores):.3f}")


# ----------------------------------------------------------
# Tab 2 — Transaction Amount
# ----------------------------------------------------------
with tab2:
    st.subheader("Transaction Amount: Fraud vs Legitimate")
    st.caption("Distribution of amounts (capped at 99th percentile to reduce skew).")

    if "TransactionAmt" in X_clean.columns:
        amt_fraud = X_clean.loc[all_labels == 1, "TransactionAmt"].values
        amt_legit = X_clean.loc[all_labels == 0, "TransactionAmt"].values
        cap       = float(np.percentile(np.concatenate([amt_fraud, amt_legit]), 99))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=amt_legit[amt_legit <= cap], name="Legitimate",
            nbinsx=60, opacity=0.65,
            marker_color="seagreen",
            histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=amt_fraud[amt_fraud <= cap], name="Fraud",
            nbinsx=60, opacity=0.75,
            marker_color="crimson",
            histnorm="probability density",
        ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Transaction Amount",
            yaxis_title="Density",
            legend=dict(orientation="h", y=1.1),
            height=380,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Median — Fraud", f"${np.median(amt_fraud):.0f}")
        cb.metric("Median — Legit", f"${np.median(amt_legit):.0f}")
        cc.metric("Mean — Fraud",   f"${np.mean(amt_fraud):.0f}")
        cd.metric("Mean — Legit",   f"${np.mean(amt_legit):.0f}")
    else:
        st.info("TransactionAmt column not found in features.")


# ----------------------------------------------------------
# Tab 3 — Threshold vs Cost
# ----------------------------------------------------------
with tab3:
    st.subheader("Threshold vs Business Cost")
    st.caption(
        f"Total cost = FN × {FN_COST} + FP × {FP_COST}. "
        "Lower is better. Orange dotted line = cost-optimal threshold."
    )

    thresholds = np.linspace(0.05, 0.95, 200)
    costs, precisions, recalls = [], [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        fn = int(((all_labels == 1) & (preds == 0)).sum())
        fp = int(((all_labels == 0) & (preds == 1)).sum())
        tp = int(((all_labels == 1) & (preds == 1)).sum())
        costs.append(fn * FN_COST + fp * FP_COST)
        precisions.append(tp / (tp + fp + 1e-9))
        recalls.append(tp / (tp + fn + 1e-9))

    optimal_t = thresholds[int(np.argmin(costs))]
    min_cost  = min(costs)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=thresholds, y=costs, mode="lines", name="Total Cost",
        line=dict(color="royalblue", width=2),
    ))
    fig_cost.add_vline(
        x=threshold, line_dash="dash", line_color="black", line_width=2,
        annotation_text=f"Current {threshold:.3f}", annotation_position="top right",
    )
    fig_cost.add_vline(
        x=optimal_t, line_dash="dot", line_color="orange", line_width=2,
        annotation_text=f"Optimal {optimal_t:.3f}", annotation_position="top left",
    )
    fig_cost.update_layout(
        xaxis_title="Decision Threshold", yaxis_title="Total Cost",
        height=360, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    ca, cb = st.columns(2)
    ca.metric("Cost-optimal threshold", f"{optimal_t:.3f}")
    cb.metric("Minimum total cost",     f"{min_cost:,}")

    # Precision / Recall tradeoff
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=thresholds, y=precisions, mode="lines", name="Precision",
        line=dict(color="seagreen", width=2),
    ))
    fig_pr.add_trace(go.Scatter(
        x=thresholds, y=recalls, mode="lines", name="Recall",
        line=dict(color="crimson", width=2),
    ))
    fig_pr.add_vline(
        x=threshold, line_dash="dash", line_color="black", line_width=2,
        annotation_text=f"Current {threshold:.3f}",
    )
    fig_pr.update_layout(
        title="Precision & Recall vs Threshold",
        xaxis_title="Decision Threshold", yaxis_title="Score",
        legend=dict(orientation="h", y=1.1),
        height=320, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_pr, use_container_width=True)


# ----------------------------------------------------------
# Tab 4 — Confusion Matrix
# ----------------------------------------------------------
with tab4:
    st.subheader("Confusion Matrix at Current Threshold")
    st.caption(f"Threshold = {threshold:.3f} | holdout set")

    preds_bin = (all_probs >= threshold).astype(int)
    tp = int(((all_labels == 1) & (preds_bin == 1)).sum())
    tn = int(((all_labels == 0) & (preds_bin == 0)).sum())
    fp = int(((all_labels == 0) & (preds_bin == 1)).sum())
    fn = int(((all_labels == 1) & (preds_bin == 0)).sum())

    fig_cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit",    "Actual Fraud"],
        text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        textfont={"size": 16},
    ))
    fig_cm.update_layout(height=360, margin=dict(t=20, b=40))
    st.plotly_chart(fig_cm, use_container_width=True)

    total = tp + tn + fp + fn
    ca, cb, cc, cd = st.columns(4)
    ca.metric("Accuracy",  f"{(tp+tn)/total:.3f}")
    cb.metric("Precision", f"{tp/(tp+fp+1e-9):.3f}")
    cc.metric("Recall",    f"{tp/(tp+fn+1e-9):.3f}")
    cd.metric("F1",        f"{2*tp/(2*tp+fp+fn+1e-9):.3f}")