import pandas as pd
from pathlib import Path

def load_clean_train(path: str = "data/clean_train.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)