import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)


def plot_confusion_matrix_argmax(
    y_true,
    y_pred,
    classes=None,
    normalize: str | None = "true",  # "true" recommended
):
    """
    Confusion matrix for an argmax classifier.
    Returns (cm_df, fig).

    normalize:
      - "true": rows sum to 1 (recommended)
      - "pred", "all", or None for counts
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax,
        values_format=".2f" if normalize else "d",
        colorbar=True,
    )
    ax.set_title("Confusion matrix" + (" (normalized by True label)" if normalize else " (counts)"))
    fig.tight_layout()

    return cm_df, fig



def plot_pr_auc_macro_from_proba(
    y_true,
    proba: np.ndarray,
    classes,
):
    """
    Compute macro-average AP (PR-AUC) for multiclass via one-vs-rest, and plot
    a macro-averaged PR curve.

    Inputs:
      y_true: array-like of shape (n,)
      proba:  ndarray of shape (n, K) with predicted probabilities
      classes: list/array of length K matching columns of proba

    Returns: (macro_ap, fig)
    """
    y_true = np.asarray(y_true).astype(str)
    proba = np.asarray(proba)
    classes = list(classes)

    if proba.ndim != 2 or proba.shape[0] != len(y_true) or proba.shape[1] != len(classes):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, proba={proba.shape}, classes={len(classes)}")

    ap_list = []
    precisions = []
    recalls = []

    for j, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0:
            continue  # class not present in eval set

        y_score = proba[:, j]
        ap_list.append(average_precision_score(y_bin, y_score))

        p, r, _ = precision_recall_curve(y_bin, y_score)
        precisions.append(p)
        recalls.append(r)

    if not ap_list:
        raise ValueError("No classes in y_true had positive examples; cannot compute PR-AUC.")

    macro_ap = float(np.mean(ap_list))

    # Macro PR curve by interpolating precision onto a common recall grid
    recall_grid = np.linspace(0, 1, 200)
    prec_on_grid = []
    for p, r in zip(precisions, recalls):
        order = np.argsort(r)
        r_sorted = r[order]
        p_sorted = p[order]
        prec_on_grid.append(
            np.interp(recall_grid, r_sorted, p_sorted, left=p_sorted[0], right=p_sorted[-1])
        )

    macro_precision = np.mean(np.vstack(prec_on_grid), axis=0)

    fig, ax = plt.subplots()
    ax.plot(recall_grid, macro_precision)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Macro PR curve (macro AP = {macro_ap:.3f})")
    fig.tight_layout()

    return macro_ap, fig
