from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "aqi_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Air Quality Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # IMPORTANT: keep feature order same as training
    feature_order = model.feature_names_in_

    input_data = pd.DataFrame([[ 
        data["PM2.5"],
        data["PM10"],
        data["NO2"],
        data["SO2"],
        data["CO"],
        data["O3"]
    ]], columns=feature_order)

    prediction = model.predict(input_data)

    return jsonify({
        "Predicted_AQI": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
