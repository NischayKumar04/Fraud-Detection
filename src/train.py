import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_loader import load_clean_train
from src.features import time_based_split
from src.utils import MODELS_DIR, ensure_dirs, save_joblib, save_json

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


def find_best_threshold_f1(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    best = {"best_f1": -1.0, "best_threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for i, th in enumerate(t):
        pp, rr = p[i], r[i]
        f1 = 0.0 if (pp + rr) == 0 else 2 * pp * rr / (pp + rr)
        if f1 > best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_threshold": float(th),
                "precision": float(pp),
                "recall": float(rr),
            }
    return best


def find_best_threshold_by_cost(y_true, y_prob, fn_cost=20.0, fp_cost=1.0, grid_size=500):
    thresholds = np.linspace(0.001, 0.999, grid_size)
    best = {
        "best_threshold": 0.5,
        "min_cost": float("inf"),
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "tp": 0,
    }

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn_cost * fn + fp_cost * fp

        if total_cost < best["min_cost"]:
            best = {
                "best_threshold": float(th),
                "min_cost": float(total_cost),
                "fn": int(fn),
                "fp": int(fp),
                "tn": int(tn),
                "tp": int(tp),
            }
    return best


def build_models(scale_pos_weight):
    models = {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=300,
                solver="saga",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
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
    parser.add_argument("--max_rows", type=int, default=0, help="0 means use full dataset")
    parser.add_argument("--target", type=str, default="isFraud")
    parser.add_argument("--threshold_mode", type=str, default="cost", choices=["f1", "cost"])
    parser.add_argument("--fn_cost", type=float, default=20.0)
    parser.add_argument("--fp_cost", type=float, default=1.0)
    args = parser.parse_args()

    ensure_dirs()
    df = load_clean_train("data/clean_train.csv")

    # optional cap for debug: keep earliest rows to preserve time order
    if args.max_rows and len(df) > args.max_rows:
        df = df.sort_values("TransactionDT").iloc[: args.max_rows].copy()

    X_train, X_valid, y_train, y_valid = time_based_split(
        df, target_col=args.target, time_col="TransactionDT", test_size=0.2
    )

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    model_map = build_models(scale_pos_weight)
    run_models = model_map if args.model == "all" else {args.model: model_map[args.model]}

    results = {}
    best_name, best_model = None, None
    best_pr_auc, best_threshold = -1.0, 0.5

    for name, model in run_models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_valid)[:, 1]

        default_metrics = compute_metrics(y_valid, prob, threshold=0.5)
        f1_search = find_best_threshold_f1(y_valid, prob)

        if args.threshold_mode == "cost":
            cost_search = find_best_threshold_by_cost(
                y_valid, prob, fn_cost=args.fn_cost, fp_cost=args.fp_cost
            )
            selected_threshold = cost_search["best_threshold"]
        else:
            cost_search = None
            selected_threshold = f1_search["best_threshold"]

        selected_metrics = compute_metrics(y_valid, prob, threshold=selected_threshold)

        results[name] = {
            "default_0.5": default_metrics,
            "threshold_search_f1": f1_search,
            "threshold_search_cost": cost_search,
            "selected_threshold_mode": args.threshold_mode,
            "selected_threshold": float(selected_threshold),
            "selected_metrics": selected_metrics,
        }

        # Keep model ranking by PR-AUC (threshold-independent)
        if selected_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = selected_metrics["pr_auc"]
            best_name = name
            best_model = model
            best_threshold = selected_threshold

    save_json(
        {
            "rows_used": int(len(df)),
            "target": args.target,
            "scale_pos_weight": scale_pos_weight,
            "models": results,
            "best_model": best_name,
            "best_threshold": float(best_threshold),
            "best_pr_auc": float(best_pr_auc),
            "split_type": "time_based",
            "threshold_mode": args.threshold_mode,
            "fn_cost": args.fn_cost,
            "fp_cost": args.fp_cost,
        },
        MODELS_DIR / "metrics.json",
    )

    save_json(
        {
            "best_model": best_name,
            "best_threshold": float(best_threshold),
            "rows_used": int(len(df)),
            "split_type": "time_based",
            "threshold_mode": args.threshold_mode,
            "fn_cost": args.fn_cost,
            "fp_cost": args.fp_cost,
        },
        MODELS_DIR / "best_model_info.json",
    )

    save_joblib(best_model, MODELS_DIR / "best_model.joblib")

    print("\nTraining complete.")
    print(f"Rows used: {len(df)}")
    print(f"Best model: {best_name}")
    print(f"Best threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    main()