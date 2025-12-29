import pandas as pd

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)

    df = df.drop_duplicates(subset=["subject", "body"]).copy()

    X = df[["subject", "body"]]
    y = df["label"]
    return X, y