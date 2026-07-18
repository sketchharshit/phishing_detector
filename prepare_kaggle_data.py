"""
prepare_kaggle_data.py

Converts the raw Kaggle "Phishing Email Detection" dataset
(https://www.kaggle.com/datasets/subhajournal/phishingemails) into the
schema the rest of the project expects: data/raw_emails.csv with columns
`email_text`, `label` (1 = phishing, 0 = legitimate).

The raw file has 3 columns (an unnamed index, "Email Text", "Email Type")
and a few real data-quality issues this script handles explicitly:

  1. 16 rows with a null Email Text - dropped (documented in the dataset
     itself and confirmed in several papers that use this data).
  2. 1,112 exact duplicate Email Text rows - dropped, keeping the first
     occurrence, so the same email isn't counted multiple times across
     train/test.
  3. One row is ~17 million characters long - almost certainly a CSV
     parsing artifact (a stray quote/comma merged several records into
     one field) rather than a real single email. More generally, a long
     tail of unusually long rows exists. Text is truncated to 20,000
     characters, which affects only a few dozen rows out of 18,650 and
     keeps TF-IDF fitting fast without throwing away any email that's
     long for a normal reason.

Run once:
    python prepare_kaggle_data.py
"""

import pandas as pd

RAW_PATH = "data/Phishing_Email_raw.csv"
OUT_PATH = "data/raw_emails.csv"
MAX_CHARS = 20_000


def clean_kaggle_data(df, max_chars=MAX_CHARS, verbose=True):
    """Apply all three cleaning steps to the raw Kaggle dataframe and
    return a new dataframe with columns `email_text`, `label`."""
    df = df.rename(columns={"Email Text": "email_text", "Email Type": "email_type"})
    df = df[["email_text", "email_type"]]

    before = len(df)
    df = df.dropna(subset=["email_text"])
    if verbose:
        print(f"dropped {before - len(df)} rows with null email_text")

    before = len(df)
    df = df.drop_duplicates(subset=["email_text"], keep="first")
    if verbose:
        print(f"dropped {before - len(df)} exact-duplicate rows")

    n_truncated = (df["email_text"].str.len() > max_chars).sum()
    df["email_text"] = df["email_text"].str.slice(0, max_chars)
    if verbose:
        print(f"truncated {n_truncated} rows to {max_chars} characters")

    df["label"] = (df["email_type"] == "Phishing Email").astype(int)
    return df[["email_text", "label"]].reset_index(drop=True)


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"loaded: {df.shape}")

    df = clean_kaggle_data(df)

    print()
    print(f"final shape: {df.shape}")
    print(df["label"].value_counts().rename({1: "phishing", 0: "legitimate"}))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
