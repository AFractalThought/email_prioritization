# Email Prioritization Demo

Train and serve an email prioritization model with a Gradio UI for interactive predictions, metrics, and dataset preview.

[Live app](https://cb98c60b53059edaf6.gradio.live/)

## Table of Contents

[About](#about)

[Quickstart](#quickstart)

[Repo Structure](#repo-structure)

[Data](#data)

[Evaluation](#evaluation)

## About

This project includes:

- a simple email prioritization model (training + inference)
- an event-style predictor that assigns a label to an email
- a Gradio web app that supports:
  1. pasting an email (subject/body) to view the predicted label, confidence, probabilities, and reasons
  2. viewing evaluation metrics on a fixed test split (confusion matrix + macro PR-AUC)
  3. previewing the dataset used for evaluation

## Quickstart

### UI

Use the **Simulate email event** tab in the [Gradio app](https://cb98c60b53059edaf6.gradio.live/): paste an email `subject` + `body` to get a prediction.

### Local Configuration

 > Note: `src/app.py` may use `demo.launch(share=True)` for public demos. For local-only runs, set `share=False` or remove the argument.

#### Prereqs
- Python 3.11+ (tested on Python 3.12)
  
- Supabase project credentials in a local `.env` file (see `.env.example`).

The Gradio app entrypoint is `src/app.py`. Run it locally with:

```bash
pip install -r requirements.txt
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

## Data

- [Public source dataset](https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier) used for training
  - See details in [Writeup: dataset selection](writeup.md#dataset-selection)
  - View dataset preview in **Dataframe** tab in [Gradio app](https://cb98c60b53059edaf6.gradio.live/)
- For the demo app, the processed dataset is mirrored into Supabase and trained model is stored in Supabase Storage 
  - See details in [Writeup: storage selection](writeup.md#storage-and-deployment-choices)

## Evaluation

- View performance metrics in the **Performance** tab of the [Gradio app](https://cb98c60b53059edaf6.gradio.live/).
- Evaluation uses a seeded **stratified train/test split** 
  - PR-AUC (Average Precision)** + PR curve (threshold-independent)
  - **Confusion matrix (counts)** under the **argmax** decision policy
- More details: [Writeup: Evaluation](writeup.md#evaluation)