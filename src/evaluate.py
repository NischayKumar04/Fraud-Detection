import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_clean_train
from src.features import split_xy
from src.utils import load_joblib, MODELS_DIR


def main(default_threshold: float = 0.5):
    df = load_clean_train("data/clean_train.csv")
    X, y = split_xy(df, target_col="isFraud")
    X = X.astype("float32")

    model = load_joblib(MODELS_DIR / "best_model.joblib")
    prob = model.predict_proba(X)[:, 1]

    threshold = default_threshold
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            threshold = float(info.get("best_threshold", default_threshold))

    pred = (prob >= threshold).astype(int)

    print(f"Using threshold: {threshold:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, pred))
    print("\nClassification Report:")
    print(classification_report(y, pred, digits=4))


if __name__ == "__main__":
    main(0.5)