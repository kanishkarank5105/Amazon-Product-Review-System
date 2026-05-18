"""
app/app.py
----------
Flask web app for sentiment prediction.

Endpoints:
    GET  /            -> HTML form
    POST /predict     -> form submission, renders result on the page
    POST /api/predict -> JSON API for programmatic access

Run from the project root:
    python app/app.py
or
    flask --app app/app.py run
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Make project root importable so we can use sentiment_predict + preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentiment_predict import load_model  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the logistic model at startup (lightweight, no GPU).
# Switch to "distilbert" if you've trained that model and want higher accuracy.
DEFAULT_MODEL = os.environ.get("SENTIMENT_MODEL", "logistic")

try:
    MODEL = load_model(DEFAULT_MODEL)
    print(f"[INFO] Loaded sentiment model: {DEFAULT_MODEL}")
except FileNotFoundError as e:
    MODEL = None
    print(f"[WARN] {e}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, review="")


@app.route("/predict", methods=["POST"])
def predict():
    review = (request.form.get("review") or "").strip()
    if not review:
        return render_template(
            "index.html",
            result={"error": "Please enter a review."},
            review="",
        )
    if MODEL is None:
        return render_template(
            "index.html",
            result={"error": "Model not loaded. Run training first."},
            review=review,
        )
    result = MODEL.predict(review)
    return render_template("index.html", result=result, review=review)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()
    if not review:
        return jsonify({"error": "Missing 'review' in request body"}), 400
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 503
    return jsonify(MODEL.predict(review))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None, "model": DEFAULT_MODEL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)