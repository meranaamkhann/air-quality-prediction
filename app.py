"""
Air Quality Prediction API
---------------------------
A small Flask API that serves AQI (Air Quality Index) predictions from a
pre-trained Random Forest model.

Endpoints:
    GET  /          -> basic service info
    GET  /health     -> health check (used by deployment platforms)
    POST /predict    -> predict AQI from pollutant readings
"""

import logging
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "aqi_model.pkl")

model = None
FEATURE_ORDER = []


def load_model():
    """Load the pickled model and remember the feature order it expects."""
    global model, FEATURE_ORDER

    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)

    model = loaded_model
    FEATURE_ORDER = list(model.feature_names_in_)
    logger.info("Model loaded successfully. Expected features: %s", FEATURE_ORDER)


try:
    load_model()
except FileNotFoundError:
    logger.error("Model file not found at %s. /predict will be unavailable until it exists.", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model from %s", MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "service": "Air Quality Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health_check": "GET /health",
            "predict": "POST /predict",
        },
    })


@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy", "model_loaded": True}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server. Please try again later."}), 503

    if not request.is_json:
        return jsonify({"error": "Request body must be JSON with Content-Type: application/json"}), 415

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    # Check every required feature is present
    missing_fields = [f for f in FEATURE_ORDER if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required field(s)",
            "missing_fields": missing_fields,
            "required_fields": FEATURE_ORDER,
        }), 400

    # Check every value is numeric
    values = []
    invalid_fields = []
    for feature in FEATURE_ORDER:
        value = data[feature]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            invalid_fields.append(feature)

    if invalid_fields:
        return jsonify({
            "error": "Field(s) must be numeric",
            "invalid_fields": invalid_fields,
        }), 400

    try:
        input_df = pd.DataFrame([values], columns=FEATURE_ORDER)
        prediction = model.predict(input_df)[0]
    except Exception:
        logger.exception("Prediction failed for input: %s", data)
        return jsonify({"error": "Prediction failed due to an internal error"}), 500

    return jsonify({"Predicted_AQI": round(float(prediction), 2)})


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=debug_mode)
