# -*- coding: utf-8 -*-
"""
app.py - Flask web application for Crop Price Prediction

Loads the trained model from model/crop_price_model.pkl and serves
a web interface on localhost for predicting crop modal prices.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_price_model.pkl")

app = Flask(__name__)

# --- Load Model ---
print("[INFO] Loading model artifacts...")
if os.path.isfile(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    features = artifacts["features"]
    best_model_name = artifacts["best_model_name"]
    results = artifacts["results"]
    dropdown_options = artifacts["dropdown_options"]
    model_available = True
    print(f"[INFO] Loaded model: {best_model_name}")
else:
    model = None
    encoder = None
    features = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    best_model_name = "Not trained"
    results = []
    dropdown_options = {feature: [] for feature in features}
    model_available = False
    print(f"[WARNING] Model not found at {MODEL_PATH}. Train it after adding data/Dataset.csv.")


# --- Routes ---
@app.route("/")
def index():
    """Home page with the prediction form."""
    return render_template(
        "index.html",
        dropdown_options=dropdown_options,
        model_name=best_model_name,
        results=results,
        model_available=model_available,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint: accepts form data, returns predicted price."""
    if not model_available:
        return jsonify({
            "success": False,
            "error": "The prediction model is not trained. Add data/Dataset.csv and run train_model.py.",
        }), 503

    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        sample = pd.DataFrame([{
            "State": data["state"],
            "District": data["district"],
            "Market": data["market"],
            "Commodity": data["commodity"],
            "Variety": data["variety"],
            "Grade": data["grade"],
        }])

        sample_encoded = encoder.transform(sample)
        pred_log = model.predict(sample_encoded)
        predicted_price = round(float(np.expm1(pred_log)[0]), 2)

        return jsonify({
            "success": True,
            "predicted_price": predicted_price,
            "model_used": best_model_name,
            "inputs": data,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/about")
def about():
    """About page with model details."""
    return render_template(
        "about.html",
        model_name=best_model_name,
        results=results,
    )


if __name__ == "__main__":
    print("\nCrop Price Predictor is running!")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
