# Email Prioritization Demo
Train and serve an email prioritization model with a Gradio UI for interactive predictions, metrics, and dataset preview.

[Live app](https://cb98c60b53059edaf6.gradio.live/)


## Table of Contents

[About](#about)

[Data](#data)

[Repo Structure](#repostructure)

[Usage](#usage)

## About

This project includes:

- a simple email prioritization model (training + inference)
- an event-style predictor that assigns a label to an email
- a Gradio web app that supports:
  1. pasting an email (subject/body) to view the predicted label, confidence, probabilities, and reasons
  2. viewing evaluation metrics on a fixed test split (confusion matrix + macro PR-AUC)
  3. previewing the dataset used for evaluation


## Data & Model Artifacts

- Public source dataset (used to create labels): https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier
- For the demo app, the processed dataset is mirrored into Supabase (table: `emails_labeled`).
- The trained model is stored in Supabase Storage
- [See dataset selection details](writeup.md#datasetselection)

## Usage

### UI
Use the **Simulate email event** tab in the Gradio app: paste an email `subject` + `body` to get a prediction.


### Local Configuration
This app requires Supabase credentials via environment variables (see `.env.example`).
> Note: `src/app.py` may use `demo.launch(share=True)` for public demos. For local-only runs, set `share=False` or remove the argument.

The Gradio app entrypoint is `src/app.py`. Run it locally with:

```bash
python3 src/app.py 
```













## Repo Structure

```text
src/
  app.py                 # Gradio UI
  ml/
    train.py             # model construction / training helpers
    predict.py           # prediction + reasoning helpers
    eval.py              # confusion matrix + PR-AUC plotting
    pre_process.py       # preprocessing fcns
  store/
    supabase_io.py       # Supabase fetch/upload/download utilities

```
