import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from pathlib import Path
import joblib
from ml import pre_process

def fit(df, test_size = 0.25) -> Pipeline:

    # Preprocess table (de-duplicate etc)
    X, y = pre_process.prepare_xy(df)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=37, stratify=y
        )

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

    pipe = Pipeline([
        ("text", text),
        ("clf", clf),
    ])

    return pipe.fit(X_train, y_train)

def save_model_local(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path