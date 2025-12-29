from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.model_selection import train_test_split

from store import supabase_io
from ml import predict, evaluate, pre_process

# Load env once at startup
load_dotenv(find_dotenv())


# ----------------------------
# CACHED LOADERS
# ----------------------------
@lru_cache(maxsize=1)
def get_model():
    lp = supabase_io.download_artifact(
        bucket="models",
        object_path="resend/v1/pipeline.joblib",
        local_path=".cache/pipeline.joblib",
        force=False,  # set True only if you overwrite the same remote object_path
    )
    return joblib.load(lp)

@lru_cache(maxsize=1)
def get_raw_df(limit: int = 20000) -> pd.DataFrame:
    return supabase_io.fetch_df("emails_labeled", limit=limit)


@lru_cache(maxsize=1)
def get_eval_split():
    """
    Fixed split so your performance tab is stable / reproducible.
    """
    df = get_raw_df()
    X, y = pre_process.prepare_xy(df)  # your dedupe + fillna + type coercion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=37, stratify=y
    )
    return X_train, X_test, y_train, y_test

# ----------------------------
# TAB 1: SIMULATE EMAIL EVENT
# ----------------------------
def ui_predict_one(subject: str, body: str):
    model = get_model()

    label, confidence, probs, reasons = predict.predict_one_with_reasons(
        model, subject=subject, body=body, top_k=10
    )

    # probs can be pd.Series or dict; normalize to dict for gr.Label
    if hasattr(probs, "to_dict"):
        probs_dict = probs.to_dict()
    else:
        probs_dict = dict(probs)

    # Reasons -> markdown bullets
    reasons_md = "\n".join([f"- {r}" for r in reasons]) if reasons else "_No strong features found._"

    return (
        str(label),
        float(confidence),
        probs_dict,       # for gr.Label (bars)
        reasons_md,
    )

# ----------------------------
# TAB 2: PERFORMANCE
# ----------------------------
def ui_run_eval():
    model = get_model()
    _, X_test, _, y_test = get_eval_split()

    y_hat, proba = predict.probabilities_and_labels(model, X_test)

    classes = list(model.named_steps["clf"].classes_)

    # ensure proba is numpy array for PR code
    if isinstance(proba, pd.DataFrame):
        proba_np = proba[classes].to_numpy()
    else:
        proba_np = proba

    cm_df, cm_fig = evaluate.plot_confusion_matrix_argmax(
        y_test, y_hat, classes=classes, normalize=None
    )
    pr_auc, pr_fig = evaluate.plot_pr_auc_macro_from_proba(
        y_test, proba_np, classes=classes
    )

    return cm_df, cm_fig, float(pr_auc), pr_fig


# ----------------------------
# TAB 3: DATAFRAME DISPLAY
# ----------------------------
def ui_show_df(n_rows: int):
    df = get_raw_df()
    return df.head(int(n_rows))


# ----------------------------
# APP
# ----------------------------
with gr.Blocks(title="Email Classifier Demo") as demo:
    gr.Markdown(
        """
# Email Classifier

**Data source:** [jason23322/high-accuracy-email-classifier](https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier)  
**License:** Apache-2.0  

For training details, evaluation methodology, and how predictions/reasons are computed, see the project **README**.
"""
    )

    with gr.Tabs():
        # ---- Tab 1
        with gr.Tab("Simulate email event"):
            subject = gr.Textbox(label="Subject", lines=1, placeholder="e.g. Verify your email")
            body = gr.Textbox(label="Body", lines=8, placeholder="Paste email body here...")

            btn = gr.Button("Predict")

            out_label = gr.Textbox(label="Prediction")
            out_conf = gr.Number(label="Confidence (max probability)")
            out_probs = gr.Label(label="Probabilities (all labels)", num_top_classes=10)
            out_reasons = gr.Markdown(label="Reasons (top features)")

            btn.click(
                fn=ui_predict_one,
                inputs=[subject, body],
                outputs=[out_label, out_conf, out_probs, out_reasons],
            )
            # ---- Tab 2
        with gr.Tab("Performance"):
            gr.Markdown("Uses a fixed train/test split of the labeled dataset.")
            run_eval = gr.Button("Run evaluation")

            cm_table = gr.Dataframe(label="Confusion matrix (counts)")
            cm_plot = gr.Plot(label="Confusion matrix plot")
            pr_score = gr.Number(label="Macro PR-AUC (Average Precision)")
            pr_plot = gr.Plot(label="Macro Precision–Recall curve")

            run_eval.click(
                fn=ui_run_eval,
                inputs=[],
                outputs=[cm_table, cm_plot, pr_score, pr_plot],
            )

        # ---- Tab 3
        with gr.Tab("Dataframe"):
            n_rows = gr.Slider(5, 500, value=50, step=5, label="Rows to display")
            show = gr.Button("Show rows")
            df_view = gr.Dataframe(label="emails_labeled (preview)", wrap=True)

            show.click(fn=ui_show_df, inputs=[n_rows], outputs=[df_view])

demo.launch(share = True)

