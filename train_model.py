# -*- coding: utf-8 -*-
"""
train_model.py - Train the crop price prediction model and save as .pkl

This script:
1. Loads Dataset.csv from the data/ folder
2. Cleans and preprocesses the data
3. Trains a Random Forest model (best performer from evaluation)
4. Saves the trained model, encoder, and metadata as .pkl files
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Try optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[INFO] xgboost not installed, skipping XGBoost model.")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[INFO] lightgbm not installed, skipping LightGBM model.")


def main():
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load Data ---
    print("[1/6] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"       Dataset shape: {df.shape}")

    # --- Clean ---
    print("[2/6] Cleaning data...")
    df.columns = df.columns.str.strip()
    df = df[df['Modal_Price'] > 0].copy()
    df['Log_Modal_Price'] = np.log1p(df['Modal_Price'])

    features = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
    target = 'Log_Modal_Price'

    X = df[features]
    y = df[target]
    y_orig = df['Modal_Price']

    # --- Split ---
    print("[3/6] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y, y_orig, test_size=0.20, random_state=42
    )

    # --- Encode ---
    print("[4/6] Target-encoding categorical features...")
    encoder = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    print(f"       Training: {X_train_encoded.shape[0]}, Test: {X_test_encoded.shape[0]}")

    # --- Train multiple models ---
    print("[5/6] Training models...")
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=16,
            min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=6,
            learning_rate=0.08, random_state=42
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=6,
            learning_rate=0.08, random_state=42, n_jobs=-1
        )
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=150, max_depth=6,
            learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1
        )

    results = []
    best_r2 = -1
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train_encoded, y_train)
        preds_log = model.predict(X_test_encoded)
        preds_orig = np.expm1(preds_log)

        mae = mean_absolute_error(y_test_orig, preds_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
        r2 = r2_score(y_test_orig, preds_orig)
        mape = np.mean(np.abs((y_test_orig.values - preds_orig) / y_test_orig.values)) * 100

        results.append({
            "Model": name, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        })
        print(f"       {name:25s} -> R2={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    best_model = models[best_model_name]

    # --- Extract unique values for dropdowns ---
    dropdown_options = {}
    for col in features:
        dropdown_options[col] = sorted(df[col].dropna().unique().tolist())

    # --- Save everything as pkl ---
    print(f"\n[6/6] Saving best model ({best_model_name}) and artifacts as .pkl ...")
    artifacts = {
        "model": best_model,
        "encoder": encoder,
        "features": features,
        "best_model_name": best_model_name,
        "results": results,
        "dropdown_options": dropdown_options,
    }

    pkl_path = os.path.join(MODEL_DIR, "crop_price_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"       Saved -> {pkl_path}")

    # Also save results separately for the website
    results_path = os.path.join(MODEL_DIR, "model_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"       Saved -> {results_path}")

    print("\nDone! Model trained and saved successfully.")
    print(f"   Best model: {best_model_name} (R2 = {best_r2:.4f})")
    print("   Now run: python app.py")


if __name__ == "__main__":
    main()
