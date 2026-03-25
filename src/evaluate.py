import json
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_clean_train
from src.features import time_based_split
from src.utils import load_joblib, MODELS_DIR


def main(default_threshold: float = 0.5):
    df = load_clean_train("data/clean_train.csv")
    _, X_test, _, y_test = time_based_split(df, target_col="isFraud", time_col="TransactionDT", test_size=0.2)
    X_test = X_test.astype("float32")

    model = load_joblib(MODELS_DIR / "best_model.joblib")
    prob = model.predict_proba(X_test)[:, 1]

    threshold = default_threshold
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            threshold = float(info.get("best_threshold", default_threshold))

    pred = (prob >= threshold).astype(int)

    print(f"Using threshold: {threshold:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main(0.5)