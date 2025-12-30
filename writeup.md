# Data Scientist Trust & Safety Take Home Challenge — Write Up

This document summarizes the key implementation decisions behind the submission: dataset selection and label mapping, model/features, confidence + reasoning definitions, evaluation choices, and storage/deployment setup.

## Dataset selection

### Why I did not use the provided dataset

I initially explored the dataset included in the prompt, but it introduced several issues that would have made the model and evaluation misleading:

- **Weak feature coherence:** In many rows, the `From`, `Subject`, and `Body` fields did not appear semantically consistent (suggesting independent templating). This makes it difficult to learn realistic correlations across fields and undermines the intent of using all three signals.
  
- **Missing “priority” examples:** There were few/no clear MFA or verification-code examples, which is explicitly required for the `Prioritize` class.
  
- **Label mismatch with task:** Labels were `ham/spam`, while the assignment requires `priority/default/slow`. I could have generated new labels, but under time constraints it was more reliable to use a dataset whose categories already aligned with the goal.

### Why I chose the Hugging Face dataset

I used a [a dataset obtained from Hugging Face](https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier) `jason23322/high-accuracy-email-classifier` (Apache-2.0):

- The dataset’s categories map directly to the assignment’s required classes (especially verification codes).
  
- The examples are more internally coherent (subject/body alignment is generally consistent).
  
- Duplication exists (as in most datasets) but is manageable; I explicitly **de-duplicate identical subject/body pairs** before splitting to reduce leakage risk.

### Caveats / limitations of the chosen dataset

- The dataset is still **simulated**, so performance is not representative of real email traffic.
  
- It does **not include a sender field**. I realize the assignment specifically mentions using `from` (available in the provided dataset). I chose not to fabricate sender values or mix in another dataset due to scope/complexity, and because subject/body alone were sufficient to demonstrate the approach.

## Label mapping

### Original dataset labels

`['promotions', 'spam', 'social_media', 'forum', 'verify_code', 'updates']`

### Mapping to assignment labels

| Assignment label | Description (prompt)              | Dataset categories mapped         |
|---|---|---|
| **Prioritize** | MFA codes / verification           | `verify_code`                     |
| **Default**    | Generic messages                   | `social_media`, `forum`, `updates`|
| **Slow**       | Non-urgent promotional messages    |`promotions`
| **(Spam*)**       | NOT included in assignment    | `spam`                      |

### Why I included `spam` as an extra class

The assignment defines three labels, but the dataset contains a clear spam category. I kept `spam` as a separate label instead of dropping it or mapping it into `slow`, because in a realistic pipeline you’d rather route obvious spam away from the inbox than treat it as “slow.” This makes the demo more practical, and the model still outputs probabilities across all classes.

## Model approach

### Vectorization

I used **TF–IDF** features for the `subject` and `body`.

- I considered embeddings/LLM-based approaches, but TF–IDF is a strong baseline, easy to debug, and well-suited to short templated text like verification codes and promotions.
  
- I intentionally vectorize **subject and body separately** (two TF–IDF vectorizers inside a `ColumnTransformer`). This preserves the distinction between words appearing in the subject vs the body (e.g., “verify” in a subject line is often more predictive than in a long email body).

### Model choice

I used **multinomial Logistic Regression**.

- Logistic regression is interpretable, fast, and a good baseline for text classification.
  
- I did not run hyperparameter tuning (e.g., grid search) under time constraints; the goal was a clean, understandable pipeline and working demo or explore other model choices

### Decision policy

The model outputs probabilities for each label via `predict_proba`.

- The default policy is **argmax**: choose the label with the highest probability.
  
- In a production system, you might use **class-specific thresholds** (e.g., prioritize high recall for `Prioritize`, high precision for `Spam`). I kept argmax to keep behavior simple and evaluation straightforward.

## Confidence and reasoning

### Confidence

**Confidence = max predicted probability** for the selected label (i.e., `max(p(label | email))`).

This is a standard definition for multi-class models. Note: probabilities are not guaranteed to be perfectly calibrated, but they are a useful relative measure for the demo.

### Reasoning

To produce human-readable reasons, I compute **top contributing TF–IDF features** for the predicted class by combining:

- the email’s TF–IDF vector (non-zero features)
- the logistic regression coefficients for the predicted class

This yields a ranked list of tokens/ngrams that most influenced the decision (“top features” explanation).

**Interpretability note:** The “reasons” shown in the app come from logistic regression feature contributions (TF-IDF weights × class coefficients). This approach is specific to linear models. If I were to switch to non-linear models (e.g., tree ensembles or neural embeddings), I’d replace this with a model-agnostic explanation method such as SHAP (or integrated gradients for neural models), at the cost of additional complexity and runtime.

## Evaluation

### Metrics shown in the UI

I report two complementary views:

1. **Confusion matrix (counts)** using the argmax policy  
   - Provides concrete error counts by class.
2. **Macro PR-AUC (Average Precision)** as threshold-independent performance  
   - PR-AUC is appropriate under class imbalance and aligns with “ranking quality” of `predict_proba`.

## Storage and deployment choices

### Storage / data

I used **Supabase** because:

- it provides a simple Postgres backend for storing labeled examples
  
- it includes object storage suitable for model artifacts (similar to S3)
  
- I have used it before and it integrates cleanly with Python

### App

I used **Gradio** because it supports:

- quick “send a test email event” workflows
  
- easy visualization of plots and dataframes
  
- simple deployment patterns

## How the system works end-to-end

### Training (local)

- Fetch training data from Supabase
  
- Preprocess (de-duplicate, split)
  
- Vectorize text with TF–IDF + Logistic Regression pipeline
  
- Save model artifact locally (`joblib`)
  
- Upload artifact to Supabase Storage

### App (online)

- Download the trained artifact from Supabase Storage (cached locally)
  
- Fetch the dataset from Supabase (cached)
  
- Tabs:
  - **Simulate email event:** user enters subject/body → app returns label, confidence, per-class probabilities, and reasons
  - **Performance:** computes confusion matrix + PR curve / macro PR-AUC on a fixed split
  - **Dataframe:** displays a preview of the dataset

## Notes on real-world performance and model building

The model performs well on this simulated dataset, but performance will drop on real email traffic due to:

- broader vocabulary and writing styles
  
- adversarial spam behavior
  
- distribution shifts (new templates, new senders, different user contexts)

A production-grade version would require:

- real emails
  
- better calibration + thresholding policies
  
- continuous evaluation / monitoring
  
- richer features including sender domain reputation and message metadata
  
- trying multiple models, and policies, hyperparameter tuning
