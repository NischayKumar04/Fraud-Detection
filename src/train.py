import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_clean_train
from src.features import split_xy
from src.utils import MODELS_DIR, ensure_dirs, save_joblib, save_json

# Optional imports
HAS_LGBM = True
HAS_XGB = True
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    # t is len(p)-1
    best = {"best_f1": -1.0, "best_threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for i, th in enumerate(t):
        pp = p[i]
        rr = r[i]
        f1 = 0.0 if (pp + rr) == 0 else 2 * pp * rr / (pp + rr)
        if f1 > best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_threshold": float(th),
                "precision": float(pp),
                "recall": float(rr),
            }
    return best


def build_models(scale_pos_weight):
    models = {
        "lr": LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_split=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=350,
            max_depth=8,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "lr", "rf", "lgbm", "xgb"])
    parser.add_argument("--max_rows", type=int, default=120000, help="memory-safe row cap")
    parser.add_argument("--target", type=str, default="isFraud")
    args = parser.parse_args()

    ensure_dirs()

    df = load_clean_train("data/clean_train.csv")
    X, y = split_xy(df, target_col=args.target)

    # memory-safe dtype
    X = X.astype("float32")

    # optional row cap
    if len(X) > args.max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=args.max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    model_map = build_models(scale_pos_weight)

    if args.model != "all":
        if args.model not in model_map:
            raise RuntimeError(
                f"Model '{args.model}' not available. "
                f"Install missing dependency (lightgbm/xgboost) if needed."
            )
        run_models = {args.model: model_map[args.model]}
    else:
        run_models = model_map

    results = {}
    best_name = None
    best_pr_auc = -1.0
    best_model = None
    best_threshold = 0.5

    for name, model in run_models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_valid)[:, 1]

        default_metrics = compute_metrics(y_valid, prob, threshold=0.5)
        tuned = find_best_threshold(y_valid, prob)
        tuned_metrics = compute_metrics(y_valid, prob, threshold=tuned["best_threshold"])

        results[name] = {
            "default_0.5": default_metrics,
            "threshold_search": tuned,
            "tuned": tuned_metrics,
        }

        pr_auc = tuned_metrics["pr_auc"]
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_model = model
            best_threshold = tuned["best_threshold"]

    # save outputs
    save_json(
        {
            "rows_used": int(len(X)),
            "target": args.target,
            "scale_pos_weight": scale_pos_weight,
            "models": results,
            "best_model": best_name,
            "best_threshold": float(best_threshold),
            "best_pr_auc": float(best_pr_auc),
        },
        MODELS_DIR / "metrics.json",
    )
    save_json(
        {
            "best_model": best_name,
            "best_threshold": float(best_threshold),
            "rows_used": int(len(X)),
        },
        MODELS_DIR / "best_model_info.json",
    )
    save_joblib(best_model, MODELS_DIR / "best_model.joblib")

    print("\nTraining complete.")
    print(f"Rows used: {len(X)}")
    print(f"Best model: {best_name}")
    print(f"Best threshold: {best_threshold:.4f}")
    print("Saved -> models/best_model.joblib, models/metrics.json, models/best_model_info.json")


if __name__ == "__main__":
    main()