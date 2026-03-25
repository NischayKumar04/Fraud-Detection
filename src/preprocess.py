import argparse
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from src.utils import ROOT, DATA_DIR, MODELS_DIR, ensure_dirs, save_joblib


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            df[col] = df[col].astype("float32")
        elif dt == "int64":
            cmin, cmax = df[col].min(), df[col].max()
            if cmin >= 0 and cmax <= 255:
                df[col] = df[col].astype("uint8")
            elif cmin >= -128 and cmax <= 127:
                df[col] = df[col].astype("int8")
            elif cmin >= -32768 and cmax <= 32767:
                df[col] = df[col].astype("int16")
            else:
                df[col] = df[col].astype("int32")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" in df.columns:
        df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
        df["day"] = ((df["TransactionDT"] // (3600 * 24)) % 7).astype("int8")
        df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
        df["is_fraud_peak_hour"] = df["hour"].isin([3, 4, 5]).astype("int8")
        df["is_weekend"] = df["day"].isin([5, 6]).astype("int8")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" in df.columns:
        df["log_amount"] = np.log1p(df["TransactionAmt"]).astype("float32")
        cents = (df["TransactionAmt"] - df["TransactionAmt"].fillna(0).astype(int)).round(2)
        df["amount_cents"] = cents.astype("float32")
        df["is_round_amount"] = (df["amount_cents"] == 0).astype("int8")
        df["amount_bucket"] = pd.cut(
            df["TransactionAmt"],
            bins=[0, 50, 100, 200, 500, 1000, 5000, float("inf")],
            labels=[0, 1, 2, 3, 4, 5, 6],
        ).astype("float32")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["card1", "card2", "addr1"]:
        if col not in df.columns:
            continue
        df[f"{col}_txn_count"] = df.groupby(col)[col].transform("size").fillna(0).astype("int32")
        df[f"{col}_mean_amt"] = df.groupby(col)["TransactionAmt"].transform("mean").fillna(0).astype("float32")
        df[f"{col}_amt_deviation"] = (df["TransactionAmt"] - df[f"{col}_mean_amt"]).abs().fillna(0).astype("float32")
        df[f"{col}_std_amt"] = df.groupby(col)["TransactionAmt"].transform("std").fillna(0).astype("float32")
    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"card1", "addr1"}
    if not required.issubset(df.columns):
        return df

    G = nx.Graph()

    card_counts = df["card1"].value_counts()
    active_cards = card_counts[card_counts >= 3].index
    df_graph = df[df["card1"].isin(active_cards)][["card1", "addr1"]].dropna()

    addr_groups = df_graph.groupby("addr1")["card1"].apply(set)

    for _, cards in addr_groups.items():
        cards = list(cards)
        if len(cards) < 2 or len(cards) > 50:
            continue
        for i in range(len(cards)):
            for j in range(i + 1, min(len(cards), i + 10)):
                G.add_edge(cards[i], cards[j])

    card_degree = dict(G.degree())
    card_clustering = nx.clustering(G)
    card_pagerank = nx.pagerank(G, max_iter=50, tol=1e-4)

    df["graph_degree"] = df["card1"].map(card_degree).fillna(0).astype("float32")
    df["graph_clustering"] = df["card1"].map(card_clustering).fillna(0).astype("float32")
    df["graph_pagerank"] = df["card1"].map(card_pagerank).fillna(0).astype("float32")
    df["graph_degree_log"] = np.log1p(df["graph_degree"]).astype("float32")

    return df


def add_missing_flags_and_impute(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    medians = {}
    cat_fill = {}

    for c in num_cols:
        miss_pct = df[c].isnull().mean() * 100
        if 1 < miss_pct < 90:
            df[f"{c}_was_missing"] = df[c].isnull().astype("int8")
        if df[c].isnull().sum() > 0:
            med = float(df[c].median())
            medians[c] = med
            df[c] = df[c].fillna(med)

    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            cat_fill[c] = "Unknown"
            df[c] = df[c].fillna("Unknown")

    return df, medians, cat_fill


def label_encode_all(df: pd.DataFrame):
    # simple deterministic encoding via category codes
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for c in cat_cols:
        cats = sorted(df[c].astype(str).unique().tolist())
        enc_map = {v: i for i, v in enumerate(cats)}
        encoders[c] = enc_map
        df[c] = df[c].astype(str).map(enc_map).astype("int32")

    return df, encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx_path", type=str, default=str(DATA_DIR / "train_transaction.csv"))
    parser.add_argument("--id_path", type=str, default=str(DATA_DIR / "train_identity.csv"))
    parser.add_argument("--out_path", type=str, default=str(DATA_DIR / "clean_train.csv"))
    parser.add_argument("--drop_missing_threshold", type=float, default=90.0)
    args = parser.parse_args()

    ensure_dirs()

    tx_path = Path(args.tx_path)
    id_path = Path(args.id_path)
    out_path = Path(args.out_path)

    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError("Missing train_transaction.csv or train_identity.csv in data/")

    print("Loading raw files...")
    train_tx = pd.read_csv(tx_path)
    train_id = pd.read_csv(id_path)

    print("Merging...")
    df = pd.merge(train_tx, train_id, on="TransactionID", how="left")
    del train_tx, train_id
    gc.collect()

    print("Reducing memory...")
    df = reduce_memory(df)

    print("Dropping high-missing columns...")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_pct[missing_pct > args.drop_missing_threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    print("Adding engineered features...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_velocity_features(df)
    df = add_graph_features(df)

    print("Imputing + missing flags...")
    df, medians, cat_fill = add_missing_flags_and_impute(df)

    print("Encoding categoricals...")
    df, encoders = label_encode_all(df)

    print("Final memory reduce...")
    df = reduce_memory(df)

    print(f"Saving clean dataset -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    artifacts = {
        "dropped_cols": cols_to_drop,
        "medians": medians,
        "cat_fill": cat_fill,
        "label_encoders": encoders,
        "target_col": "isFraud",
    }
    save_joblib(artifacts, MODELS_DIR / "preprocess_artifacts.joblib")

    with open(MODELS_DIR / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "output": str(out_path),
                "artifacts": str(MODELS_DIR / "preprocess_artifacts.joblib"),
            },
            f,
            indent=2,
        )

    print("✅ Preprocess complete.")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()