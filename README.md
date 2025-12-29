# Email Prioritization Demo

**Live app:** https://cb98c60b53059edaf6.gradio.live/

This repo contains:
- a trained email prioritization model (loaded from Supabase Storage)
- a Gradio app to send test email events and visualize metrics
- helper modules for prediction and evaluation

For design decisions, dataset choice, and methodology, see `writeup.md`.

---

## Repo structure

```text
src/
  app.py                 # Gradio UI
  ml/
    train.py             # model construction / training helpers
    predict.py           # prediction + (optional) reasoning helpers
    eval.py              # confusion matrix + PR-AUC plotting
    pre_process.py       # preprocessing (dedupe, X/y preparation)
  store/
    supabase_io.py       # Supabase fetch/upload/download utilities

```