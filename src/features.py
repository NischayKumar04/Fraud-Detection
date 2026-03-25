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


def time_based_split(df: pd.DataFrame, target_col: str = TARGET_COL, time_col: str = "TransactionDT", test_size: float = 0.2):
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found for time-based split")
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    X_train, y_train = split_xy(train_df, target_col=target_col)
    X_test, y_test = split_xy(test_df, target_col=target_col)
    return X_train, X_test, y_train, y_test