import pandas as pd
from src.utils import MODELS_DIR, load_joblib

def predict_batch(df: pd.DataFrame, threshold: float = 0.5):
    model = load_joblib(MODELS_DIR / "best_model.joblib")
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").copy()
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = prob
    out["fraud_prediction"] = pred
    return out