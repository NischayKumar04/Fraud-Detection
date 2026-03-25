import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.data_loader import load_clean_train
from src.features import split_xy
from src.utils import MODELS_DIR, load_joblib


def _extract_class1_shap_values(shap_values):
    vals = getattr(shap_values, "values", shap_values)

    if isinstance(vals, list):
        if len(vals) >= 2:
            return np.asarray(vals[1])  # class 1
        return np.asarray(vals[0])

    vals = np.asarray(vals)
    if vals.ndim == 2:
        return vals
    if vals.ndim == 3:
        class_idx = 1 if vals.shape[2] > 1 else 0
        return vals[:, :, class_idx]

    raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")


def main(top_n: int = 15, sample_size: int = 12000):
    df = load_clean_train("data/clean_train.csv")
    X, _ = split_xy(df, target_col="isFraud")
    X = X.astype("float32")

    model = load_joblib(MODELS_DIR / "best_model.joblib")

    n = min(sample_size, len(X))
    sample = X.sample(n=n, random_state=42)
    print(f"Using SHAP sample size: {n}")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except Exception:
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)

    class1_vals = _extract_class1_shap_values(shap_values)
    mean_abs = np.abs(class1_vals).mean(axis=0)

    imp = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False).head(top_n)

    out_csv = MODELS_DIR / "shap_top_features.csv"
    imp.to_csv(out_csv, header=["mean_abs_shap"])

    plt.figure(figsize=(10, 6))
    imp.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Features by Mean |SHAP| (Class 1, n={n})")
    plt.tight_layout()
    out_png = MODELS_DIR / "shap_top_features.png"
    plt.savefig(out_png, dpi=150)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--sample_size", type=int, default=12000)  # 10k-15k target
    args = parser.parse_args()

    main(top_n=args.top_n, sample_size=args.sample_size)