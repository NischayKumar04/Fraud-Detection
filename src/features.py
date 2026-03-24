import pandas as pd

TARGET_COL = "isFraud"
DROP_COLS = ["TransactionID"]  # avoid leakage/id dependence

def split_xy(df: pd.DataFrame, target_col: str = TARGET_COL):
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe")
    X = df.drop(columns=[target_col], errors="ignore").copy()
    y = df[target_col].astype(int).copy()
    X = X.drop(columns=DROP_COLS, errors="ignore")
    return X, y