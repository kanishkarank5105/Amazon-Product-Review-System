"""
train_logistic.py
-----------------
Train a Logistic Regression sentiment classifier on Amazon reviews.

Pipeline:
    1. Load CSV from dataset/amazon_reviews.csv
    2. Clean text using preprocessing.clean_text
    3. Convert star ratings (1–5) to binary sentiment (0 = negative, 1 = positive)
    4. Vectorize with TF-IDF (unigrams + bigrams)
    5. Train Logistic Regression
    6. Evaluate on a held-out test set
    7. Save the trained model + vectorizer to models/

Run:
    python train_logistic.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from preprocessing import clean_text, label_from_rating

# ---------- Config ----------
DATA_PATH = os.path.join("dataset", "amazon_reviews.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Column names — adjust if your CSV uses different headers
TEXT_COLUMN_CANDIDATES = ["reviewText", "review", "text", "Review", "review_text"]
RATING_COLUMN_CANDIDATES = ["overall", "rating", "Rating", "star_rating", "stars"]
LABEL_COLUMN_CANDIDATES = ["sentiment", "label", "Sentiment"]


def autodetect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_prepare(data_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Initial shape: {df.shape}")
    print(f"[INFO] Columns      : {list(df.columns)}")

    text_col = autodetect_column(df, TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise ValueError(
            f"Could not find a review text column. Expected one of {TEXT_COLUMN_CANDIDATES}"
        )

    # Prefer an existing label column; otherwise derive from rating
    label_col = autodetect_column(df, LABEL_COLUMN_CANDIDATES)
    if label_col is None:
        rating_col = autodetect_column(df, RATING_COLUMN_CANDIDATES)
        if rating_col is None:
            raise ValueError(
                "No sentiment label column and no rating column found. "
                f"Need one of {LABEL_COLUMN_CANDIDATES} or {RATING_COLUMN_CANDIDATES}."
            )
        print(f"[INFO] Deriving labels from rating column '{rating_col}'")
        df["sentiment"] = df[rating_col].apply(label_from_rating)
        df = df[df["sentiment"] != -1]
    else:
        # Normalize textual labels like 'positive'/'negative' to 1/0
        if df[label_col].dtype == object:
            mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
            df["sentiment"] = df[label_col].str.lower().map(mapping)
            df = df.dropna(subset=["sentiment"])
            df["sentiment"] = df["sentiment"].astype(int)
        else:
            df["sentiment"] = df[label_col].astype(int)

    df = df[[text_col, "sentiment"]].rename(columns={text_col: "review"})
    df = df.dropna(subset=["review"]).reset_index(drop=True)

    print("[INFO] Cleaning text — this can take a minute for large files...")
    df["clean_review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["clean_review"].str.len() > 0].reset_index(drop=True)

    print(f"[INFO] Final dataset shape: {df.shape}")
    print("[INFO] Class distribution:")
    print(df["sentiment"].value_counts())
    return df


def train(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    print("\n[INFO] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy : {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n[SAVED] Model      -> {MODEL_PATH}")
    print(f"[SAVED] Vectorizer -> {VECTORIZER_PATH}")


def main():
    df = load_and_prepare(DATA_PATH)
    train(df)


if __name__ == "__main__":
    main()