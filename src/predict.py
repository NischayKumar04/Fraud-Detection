import json
import pandas as pd
from src.utils import MODELS_DIR, load_joblib


def _load_threshold(default_threshold: float = 0.5) -> float:
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            return float(info.get("best_threshold", default_threshold))
    return default_threshold


def predict_batch(df: pd.DataFrame, threshold: float | None = None):
    model = load_joblib(MODELS_DIR / "best_model.joblib")
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").copy().astype("float32")

    if threshold is None:
        threshold = _load_threshold(0.5)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = prob
    out["fraud_prediction"] = pred
    out["decision_threshold"] = threshold
    return out