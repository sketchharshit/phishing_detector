"""
app.py

Streamlit UI for the phishing email detector. Loads the models trained by
train_model.py and classifies whatever email text a user pastes in.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix

from utils import clean_text, extract_metadata, suspicious_domain

MODELS_DIR = Path("models")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "naive_bayes": "Naive Bayes",
    "neural_network": "Neural Network (MLP)",
}


@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    models = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_LABELS}
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return tfidf, scaler, models, metrics


def predict(subject, body, model_name, tfidf, scaler, models):
    raw_text = f"{subject} {body}"
    cleaned = clean_text(raw_text)
    tfidf_vec = tfidf.transform([cleaned])

    meta = np.array([extract_metadata(raw_text)])
    meta_scaled = scaler.transform(meta)

    model = models[model_name]
    if model_name == "naive_bayes":
        features = tfidf_vec
    else:
        features = hstack([tfidf_vec, csr_matrix(meta_scaled)])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return pred, prob, meta[0]


st.set_page_config(page_title="Phishing Email Detector", page_icon="\U0001F6E1\uFE0F", layout="centered")

st.title("Phishing Email Detector")
st.caption("AI-Driven Phishing Email Detection Using NLP \u2014 IICT project")

tfidf, scaler, models, metrics = load_artifacts()

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox(
        "Classifier",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda k: MODEL_LABELS[k],
        index=0,
    )
    m = metrics[model_name]
    st.metric("Test Accuracy", f"{m['accuracy']*100:.0f}%")
    col1, col2 = st.columns(2)
    col1.metric("Precision", f"{m['precision']*100:.0f}%")
    col2.metric("Recall", f"{m['recall']*100:.0f}%")
    st.caption(f"F1-score: {m['f1']:.2f}")
    st.divider()
    st.caption("Trained on the Kaggle \"Phishing Email Detection\" dataset (~17.5k emails after cleaning).")

st.subheader("Paste an email to check")

if "sender_input" not in st.session_state:
    st.session_state["sender_input"] = ""
if "subject_input" not in st.session_state:
    st.session_state["subject_input"] = ""
if "body_input" not in st.session_state:
    st.session_state["body_input"] = ""


def load_phishing_example():
    st.session_state["sender_input"] = "support@amaz0n-verify.tk"
    st.session_state["subject_input"] = "URGENT: Your account will be suspended!"
    st.session_state["body_input"] = (
        "Dear customer, your account has been limited. Click here to verify "
        "immediately: http://amaz0n-verify.tk/login?id=123456 or you will lose "
        "access permanently."
    )


def load_legit_example():
    st.session_state["sender_input"] = "hr@company.co.in"
    st.session_state["subject_input"] = "Meeting reminder: Q3 planning at 10:00 AM"
    st.session_state["body_input"] = (
        "Hi, just a reminder that our meeting on Q3 planning is scheduled for "
        "10:00 AM today. Please review the shared document beforehand."
    )


col_a, col_b = st.columns(2)
sender = col_a.text_input("Sender address (optional)", key="sender_input", placeholder="support@amaz0n-security.net")
subject = col_b.text_input("Subject", key="subject_input", placeholder="Your account will be suspended")
body = st.text_area(
    "Body",
    key="body_input",
    height=180,
    placeholder="Dear customer, we detected unusual activity on your account. "
                "Click here to verify immediately: http://verify-account.tk/login ...",
)
st.caption(
    "Sender is shown as a heuristic hint only \u2014 the training data (Kaggle "
    "phishing-email dataset) has no sender field, so the model classifies "
    "based on subject + body text alone."
)

sample_col1, sample_col2, _ = st.columns([1, 1, 2])
sample_col1.button("Try phishing example", on_click=load_phishing_example)
sample_col2.button("Try legitimate example", on_click=load_legit_example)

if st.button("Classify email", type="primary"):
    st.button(
    " Clear All",
    on_click=lambda: st.session_state.update({
        "sender_input": "",
        "subject_input": "",
        "body_input": ""
    })
)
    if not (subject or body):
        st.warning("Enter a subject or body to classify.")
    else:
        pred, prob, meta = predict(subject, body, model_name, tfidf, scaler, models)

        if pred == 1:
            st.error(f"**Phishing** \u2014 {prob*100:.1f}% confidence")
        else:
            st.success(f"**Legitimate** \u2014 {(1-prob)*100:.1f}% confidence")

        with st.expander("Why this result? (signals detected)"):
            has_url, url_count, excl_count, upper_ratio, urgency, length = meta
            st.write(f"- Contains a URL: {'yes' if has_url else 'no'} ({int(url_count)} found)")
            st.write(f"- Urgency-keyword score: {int(urgency)}")
            st.write(f"- Exclamation marks: {int(excl_count)}")
            st.write(f"- Email length: {int(length)} characters")
            if sender:
                domain_flag = suspicious_domain(sender)
                st.write(f"- Sender domain looks suspicious: {'yes' if domain_flag else 'no'} "
                         f"*(heuristic only \u2014 not seen by the model, since the training "
                         f"data has no sender field)*")

st.divider()
st.caption(
"Developed as an academic demonstration of machine learning and natural language processing techniques for phishing email detection. Predictions are provided for educational and research purposes."
)
