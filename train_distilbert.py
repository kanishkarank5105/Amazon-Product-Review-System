"""
train_distilbert.py
-------------------
Fine-tune DistilBERT for binary sentiment classification on Amazon reviews.

Notes:
- Requires `transformers`, `datasets`, `torch`, `scikit-learn`.
- Uses raw review text (no aggressive cleaning) since transformers handle
  capitalization, punctuation, etc. via their own tokenizer.
- Saves the fine-tuned model and tokenizer to models/distilbert_sentiment/.

Run:
    python train_distilbert.py
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from preprocessing import label_from_rating

# ---------- Config ----------
DATA_PATH = os.path.join("dataset", "amazon_reviews.csv")
MODEL_DIR = os.path.join("models", "distilbert_sentiment")
MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
SEED = 42

# Column auto-detection (same as logistic script)
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

    text_col = autodetect_column(df, TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise ValueError(f"No review text column found. Expected one of {TEXT_COLUMN_CANDIDATES}")

    label_col = autodetect_column(df, LABEL_COLUMN_CANDIDATES)
    if label_col is None:
        rating_col = autodetect_column(df, RATING_COLUMN_CANDIDATES)
        if rating_col is None:
            raise ValueError("No label or rating column found.")
        df["sentiment"] = df[rating_col].apply(label_from_rating)
        df = df[df["sentiment"] != -1]
    else:
        if df[label_col].dtype == object:
            mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
            df["sentiment"] = df[label_col].str.lower().map(mapping)
            df = df.dropna(subset=["sentiment"])
            df["sentiment"] = df["sentiment"].astype(int)
        else:
            df["sentiment"] = df[label_col].astype(int)

    df = df[[text_col, "sentiment"]].rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.slice(0, 2000)  # safety cap
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    print(f"[INFO] Final shape: {df.shape}")
    print("[INFO] Class distribution:")
    print(df["sentiment"].value_counts())
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_and_prepare(DATA_PATH)

    train_df, eval_df = train_test_split(
        df, test_size=0.1, random_state=SEED, stratify=df["sentiment"]
    )

    train_ds = Dataset.from_pandas(train_df[["text", "sentiment"]].rename(columns={"sentiment": "labels"}))
    eval_ds = Dataset.from_pandas(eval_df[["text", "sentiment"]].rename(columns={"sentiment": "labels"}))

    print(f"[INFO] Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("\n[INFO] Final evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n[SAVED] Fine-tuned model -> {MODEL_DIR}")


if __name__ == "__main__":
    main()