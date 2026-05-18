"""
preprocessing.py
----------------
Text cleaning utilities used by both training scripts and the Flask app.

Steps performed by `clean_text`:
1. Lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove non-alphabetic characters (keeps spaces)
5. Tokenize and remove English stopwords
6. Lemmatize tokens
7. Rejoin to a single string
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are available
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        if resource in {"stopwords", "wordnet", "omw-1.4"}:
            nltk.data.find(f"corpora/{resource}")
        else:
            nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

STOPWORDS = set(stopwords.words("english"))
# Keep negations — they matter for sentiment
NEGATIONS = {"no", "not", "nor", "n't", "never", "none", "cannot"}
STOPWORDS = STOPWORDS - NEGATIONS

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean a single review string and return a normalized version."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags (Amazon reviews sometimes have <br />)
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove punctuation and digits — keep letters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Remove stopwords + lemmatize
    tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 1]

    return " ".join(tokens)


def label_from_rating(rating) -> int:
    """
    Convert a star rating (1–5) into a sentiment label:
        - 1, 2  -> 0 (negative)
        - 3     -> dropped (return -1, caller should filter)
        - 4, 5  -> 1 (positive)
    """
    try:
        r = int(rating)
    except (ValueError, TypeError):
        return -1
    if r <= 2:
        return 0
    if r >= 4:
        return 1
    return -1


if __name__ == "__main__":
    sample = "This product is AMAZING!!! Visit https://amazon.com <br/> I love it 10/10."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))