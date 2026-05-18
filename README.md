# Amazon Review Sentiment Analysis

Binary sentiment classification (positive / negative) on Amazon product reviews using two approaches:

1. **Logistic Regression** with TF-IDF features (fast, lightweight baseline)
2. **DistilBERT** fine-tuned classifier (higher accuracy, slower)

A simple Flask web app exposes both models through a clean UI and a JSON API.

---

## Project Structure

```
Amazon-Review-Sentiment-Analysis/
├── train_logistic.py       # Train TF-IDF + Logistic Regression
├── train_distilbert.py     # Fine-tune DistilBERT
├── preprocessing.py        # Text cleaning utilities
├── sentiment_predict.py    # Unified prediction interface + CLI
├── requirements.txt
├── README.md
│
├── dataset/
│   └── amazon_reviews.csv  # YOU provide this
│
├── models/                 # trained models saved here
│
└── app/
    ├── app.py              # Flask web app
    ├── templates/
    │   └── index.html
    └── static/
        └── style.css
```

---

## 1. Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> First run will auto-download a few small NLTK resources (stopwords, wordnet, punkt).

---

## 2. Dataset

Place your CSV at `dataset/amazon_reviews.csv`.

The training scripts auto-detect column names. Supported options:

| Purpose | Accepted column names |
|---|---|
| Review text | `reviewText`, `review`, `text`, `Review`, `review_text` |
| Star rating | `overall`, `rating`, `Rating`, `star_rating`, `stars` |
| Label (optional) | `sentiment`, `label`, `Sentiment` |

If only ratings are present, they’re mapped to binary labels:
- 1–2 stars → **negative (0)**
- 3 stars → dropped (ambiguous)
- 4–5 stars → **positive (1)**

You can get a dataset from Kaggle (search “Amazon reviews”) or from
[https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html).

---

## 3. Train

### Logistic Regression (recommended first)

```bash
python train_logistic.py
```

Outputs:
- `models/logistic_model.pkl`
- `models/tfidf_vectorizer.pkl`

Should reach ~88–92% accuracy on a balanced sample.

### DistilBERT (optional, needs GPU for speed)

```bash
python train_distilbert.py
```

Outputs:
- `models/distilbert_sentiment/` — fine-tuned model + tokenizer

Should reach ~93–96% accuracy. CPU training is slow; use a subset of the data
or run on Google Colab if no GPU is available.

---

## 4. Predict from the command line

```bash
python sentiment_predict.py "I absolutely loved this product"
python sentiment_predict.py --model distilbert "Total waste of money"
```

---

## 5. Run the Web App

From the project root:

```bash
python app/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

To use the DistilBERT model in the app:

```bash
# macOS / Linux
SENTIMENT_MODEL=distilbert python app/app.py

# Windows (PowerShell)
$env:SENTIMENT_MODEL="distilbert"; python app/app.py
```

### JSON API

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"review": "This is the best purchase I have ever made"}'
```

Response:

```json
{
  "label": "Positive",
  "confidence": 0.9821,
  "model": "logistic",
  "cleaned": "best purchase ever made"
}
```

---

## 6. How it works

**Logistic pipeline**
1. Clean review (lowercase, strip HTML/URLs, remove punctuation, lemmatize, drop stopwords — but keep negations like "not").
2. Convert to TF-IDF features (unigrams + bigrams, 20k features).
3. Train logistic regression with balanced class weights.

**DistilBERT pipeline**
1. Use raw review text (transformer tokenizer handles normalization).
2. Tokenize with the DistilBERT tokenizer (max length 128).
3. Fine-tune `distilbert-base-uncased` for 2 epochs.
4. Use the trained model with a softmax head for prediction.

---

## 7. Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: dataset/amazon_reviews.csv` | Put your CSV in the `dataset/` folder. |
| `Could not find a review text column` | Rename your text column to one of the supported names (see Dataset section). |
| NLTK download error | Run `python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"` |
| DistilBERT out of memory | Lower `BATCH_SIZE` in `train_distilbert.py` (try 8 or 4). |
| App says "Model not loaded" | Run `python train_logistic.py` first. |

---

## License

MIT — use freely for learning, projects, and portfolios.