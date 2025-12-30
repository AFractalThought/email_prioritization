from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import Tuple

def probabilities_and_labels(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = model.named_steps["clf"].classes_
    proba_df = pd.DataFrame(probabilities, columns=classes, index=X.index)
    return predictions, proba_df

def predict_one(model: Pipeline, subject: str, body: str) -> tuple[str, pd.Series]:
    X = pd.DataFrame([{
        "subject": subject or "",
        "body": body or "",
    }])
    y_pred, proba_df = probabilities_and_labels(model, X)
    return y_pred[0], proba_df.iloc[0]


def explain_linear_top_features(
    model: Pipeline,
    X: pd.DataFrame,
    class_label: str,
    top_k: int = 10,
) -> list[str]:
    """
    Returns top_k feature contributions for `class_label` on the first row of X.
    Contributions are in logit units (not probability).
    """
    text = model.named_steps["text"]
    clf = model.named_steps["clf"]

    Xv = text.transform(X)  # sparse (1, n_features)
    feature_names = text.get_feature_names_out()
    classes = list(clf.classes_)
    class_idx = classes.index(class_label)

    coef = clf.coef_  # (K, n_features) for multiclass, or (1, n_features) for binary
    if coef.shape[0] == 1 and len(classes) == 2:
        # binary special-case: sklearn stores only coef for classes_[1]
        coef_c = coef[0] if class_idx == 1 else -coef[0]
    else:
        coef_c = coef[class_idx]

    row = Xv[0]
    idx = row.indices
    vals = row.data
    if idx.size == 0:
        return ["No nonzero TF-IDF features (empty subject/body after preprocessing)."]

    contrib = vals * coef_c[idx]  # per-feature contribution for this class
    order = np.argsort(np.abs(contrib))[::-1][:top_k]

    reasons = []
    for k in order:
        feat = feature_names[idx[k]]  # e.g. "subject__password" or "body__unsubscribe"
        reasons.append(f"{feat}: {contrib[k]:+.3f}")
    return reasons


def predict_one_with_reasons(model: Pipeline, subject: str, body: str, top_k: int = 10):
    X = pd.DataFrame([{"subject": subject or "", "body": body or ""}])

    proba = model.predict_proba(X)[0]
    classes = list(model.named_steps["clf"].classes_)
    probs = pd.Series(proba, index=classes)

    label = probs.idxmax()
    confidence = float(probs.max())

    reasons = explain_linear_top_features(model, X, class_label=label, top_k=top_k)
    return label, confidence, probs, reasons

   