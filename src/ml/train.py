import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pathlib import Path
import joblib


def build_pipeline() -> Pipeline:
    text = ColumnTransformer(
        transformers=[
            ("subject", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000), "subject"),
            ("body",    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000), "body"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    return Pipeline([
        ("text", text),
        ("clf", clf),
    ])

def save_model_local(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
