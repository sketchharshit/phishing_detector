"""
train_model.py

Trains all four classifiers on data/raw_emails.csv and saves everything
app.py needs at inference time: the fitted TF-IDF vectorizer, the metadata
scaler, and each trained model.

Run this once (or whenever raw_emails.csv changes) before running the app:
    python train_model.py
"""

import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack, csr_matrix

from utils import clean_text, extract_metadata, METADATA_COLS

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    df = pd.read_csv("data/raw_emails.csv")
    log(f"loaded data: {df.shape}", t0)

    df["clean_text"] = df["email_text"].apply(clean_text)
    log("cleaned text", t0)

    meta_features = df["email_text"].apply(extract_metadata)
    meta_df = pd.DataFrame(meta_features.tolist(), columns=METADATA_COLS)
    df = pd.concat([df, meta_df], axis=1)
    log("extracted metadata features", t0)

    X_text, y = df["clean_text"], df["label"]
    X_train_text, X_test_text, y_train, y_test, idx_train, idx_test = train_test_split(
        X_text, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    log(f"fit TF-IDF: {X_train_tfidf.shape}", t0)

    scaler = StandardScaler()
    meta_train = scaler.fit_transform(df.loc[idx_train, METADATA_COLS].values)
    meta_test = scaler.transform(df.loc[idx_test, METADATA_COLS].values)

    X_train_combined = hstack([X_train_tfidf, csr_matrix(meta_train)])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(meta_test)])
    log(f"combined matrix ready: train={X_train_combined.shape}, test={X_test_combined.shape}", t0)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "naive_bayes": MultinomialNB(),
        "neural_network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                          random_state=RANDOM_STATE, early_stopping=True),
    }

    # Smaller grid than before - this dataset is ~17x bigger so each fit costs more
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [None, 30], "min_samples_split": [2, 5]},
        cv=3, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train_combined, y_train)
    models["random_forest"] = grid.best_estimator_
    log(f"random forest grid search done, best params: {grid.best_params_}", t0)

    metrics = {}
    for name, model in models.items():
        if name == "naive_bayes":
            model.fit(X_train_tfidf, y_train)
            preds = model.predict(X_test_tfidf)
        else:
            model.fit(X_train_combined, y_train)
            preds = model.predict(X_test_combined)

        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
        }
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        log(f"trained {name}: {metrics[name]}", t0)

    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log("saved vectorizer, scaler, 4 models and metrics.json to models/", t0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
