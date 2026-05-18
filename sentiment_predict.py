"""
sentiment_predict.py
--------------------
Unified prediction interface for both the Logistic Regression and the
DistilBERT models. Used by app/app.py and can also be run from the CLI:

    python sentiment_predict.py "I loved this product"
    python sentiment_predict.py --model distilbert "This is terrible"
"""

import os
import sys
import argparse
from typing import Dict

import joblib

from preprocessing import clean_text

# ---------- Paths ----------
LOGISTIC_MODEL_PATH = os.path.join("models", "logistic_model.pkl")
TFIDF_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
DISTILBERT_DIR = os.path.join("models", "distilbert_sentiment")

LABEL_MAP = {0: "Negative", 1: "Positive"}


class LogisticSentimentModel:
    """TF-IDF + Logistic Regression."""

    def __init__(self, model_path=LOGISTIC_MODEL_PATH, vectorizer_path=TFIDF_PATH):
        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
            raise FileNotFoundError(
                f"Logistic model files not found. "
                f"Run `python train_logistic.py` first.\n"
                f"  expected: {model_path}\n  expected: {vectorizer_path}"
            )
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> Dict:
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "label": "Neutral",
                "confidence": 0.0,
                "model": "logistic",
                "cleaned": cleaned,
            }
        vec = self.vectorizer.transform([cleaned])
        pred = int(self.model.predict(vec)[0])
        proba = float(self.model.predict_proba(vec)[0][pred])
        return {
            "label": LABEL_MAP[pred],
            "confidence": round(proba, 4),
            "model": "logistic",
            "cleaned": cleaned,
        }


class DistilBertSentimentModel:
    """Fine-tuned DistilBERT classifier."""

    def __init__(self, model_dir=DISTILBERT_DIR):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"DistilBERT model directory not found at {model_dir}.\n"
                f"Run `python train_distilbert.py` first."
            )
        # Lazy import — only load heavy deps if this model is requested
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        if not text or not text.strip():
            return {"label": "Neutral", "confidence": 0.0, "model": "distilbert", "cleaned": ""}
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = self.torch.softmax(outputs.logits, dim=-1)[0]
            pred = int(self.torch.argmax(probs).item())
            confidence = float(probs[pred].item())
        return {
            "label": LABEL_MAP[pred],
            "confidence": round(confidence, 4),
            "model": "distilbert",
            "cleaned": text.strip(),
        }


# ---------- Loader with graceful fallback ----------
def load_model(name: str = "logistic"):
    """
    Load the requested model. If 'distilbert' is requested but unavailable,
    fall back to logistic and warn.
    """
    name = name.lower()
    if name == "distilbert":
        try:
            return DistilBertSentimentModel()
        except Exception as e:
            print(f"[WARN] DistilBERT unavailable ({e}). Falling back to logistic.")
            return LogisticSentimentModel()
    return LogisticSentimentModel()


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a single review.")
    parser.add_argument("text", nargs="+", help="Review text to classify")
    parser.add_argument(
        "--model",
        choices=["logistic", "distilbert"],
        default="logistic",
        help="Which model to use (default: logistic)",
    )
    args = parser.parse_args()

    review = " ".join(args.text)
    model = load_model(args.model)
    result = model.predict(review)
    print("\nReview      :", review)
    print("Model       :", result["model"])
    print("Sentiment   :", result["label"])
    print("Confidence  :", result["confidence"])


if __name__ == "__main__":
    main()