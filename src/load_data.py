import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, compression="gzip")
    df["text"] = df["text"].astype(str)
    return df
