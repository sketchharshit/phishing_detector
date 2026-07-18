# AI-Driven Phishing Email Detection Using NLP

Detects phishing emails from a combination of text (TF-IDF) and metadata
(URLs, exclamation marks, urgency language, etc.) using four classifiers:
Logistic Regression, Random Forest, Naive Bayes, and a small neural network.

Trained on the Kaggle ["Phishing Email Detection"](https://www.kaggle.com/datasets/subhajournal/phishingemails)
dataset (~18.6k emails, ~17.5k after cleaning).

## Project structure

```
.
├── app.py                          Streamlit app (live classification)
├── prepare_kaggle_data.py           Cleans the raw Kaggle CSV -> data/raw_emails.csv
├── train_model.py                  Trains all 4 models, saves to models/
├── utils.py                        Shared text-cleaning / feature functions
├── generate_dataset.py             Older synthetic-data generator (see Notes)
├── requirements.txt
├── data/
│   ├── Phishing_Email_raw.csv      Raw file as downloaded from Kaggle
│   └── raw_emails.csv              Cleaned: email_text, label
├── models/                         Created by train_model.py
│   ├── tfidf_vectorizer.joblib
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib        ~37 MB - see Notes
│   ├── naive_bayes.joblib
│   ├── neural_network.joblib
│   └── metrics.json
├── outputs/                        Charts saved by the notebook
└── Phishing_Email_Detection_NLP.ipynb
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. only needed if data/raw_emails.csv doesn't exist yet or you've
#    replaced data/Phishing_Email_raw.csv with a newer download
python prepare_kaggle_data.py

# 2. (re)train the models - takes ~5 minutes on this dataset size,
#    most of it in the Random Forest grid search
python train_model.py

# 3. launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Paste in a subject and body (and
optionally a sender, though it's informational only — see Notes), or click
one of the two example buttons, then hit **Classify email**.

The notebook is separate from the app — it's the write-up with EDA, all four
models, evaluation charts, and feature-importance analysis, and it takes a
similar few minutes to run end to end for the same reason. It doesn't need
to be run before the app; both import from `utils.py` and
`prepare_kaggle_data.py` so they can't drift apart, but `train_model.py` is
the one that actually produces the `models/` folder the app loads.

## Deploying for free (Streamlit Community Cloud)

1. Push this whole folder to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. **New app** → pick the repo/branch → set **Main file path** to `app.py`.
4. Deploy. First build takes a couple of minutes.

Community Cloud installs from `requirements.txt` automatically. Because
`models/` is committed alongside the code, the app loads the already-trained
models directly — it does not retrain on startup. If you change the data,
run `python train_model.py` locally first (~5 min) and commit the updated
`models/` folder before redeploying.

## Using a different dataset

Replace `data/raw_emails.csv` with any CSV that has `email_text` and `label`
columns (`label` = `1` for phishing, `0` for legitimate), then re-run
`python train_model.py`. If you're working from a fresh Kaggle download,
put the raw file at `data/Phishing_Email_raw.csv` and run
`prepare_kaggle_data.py` first — it handles null rows, exact duplicates, and
a few pathologically long rows in the raw file.

## Notes

- **This dataset's "phishing" class is broader than modern
  credential-phishing.** It's compiled from older spam/fraud corpora, so a
  lot of it is generic spam and scam email rather than "click here to
  verify your account" style messages — only about a third contain a URL at
  all, and about 1 in 20 uses urgency language. The notebook checks this
  directly rather than assuming it, and the model was tested against a
  modern-phishing-style example specifically because of it (see the
  notebook's prediction-function section).
- **No sender field.** Unlike the project's original synthetic dataset,
  this real one is just raw email text — no separate sender/subject
  columns. `suspicious_domain()` in `utils.py` is still there and the app
  still shows it as an informational hint if you type in a sender, but it
  isn't one of the features any trained model has actually seen.
- **`random_forest.joblib` is ~37 MB** (200 trees, unlimited depth, on
  ~14k training rows). Still well under GitHub's 100 MB per-file limit and
  fine for Streamlit Community Cloud, just noting it since it's much bigger
  than the other four files combined.
- `generate_dataset.py` is kept from an earlier version of this project
  that used a custom synthetic dataset instead of real data. It still runs
  standalone if useful for comparison, but nothing in the current pipeline
  depends on it.
