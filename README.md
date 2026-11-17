# 🌾 Crop Price Predictor

A Machine Learning web application that predicts the **Modal Price (₹/Quintal)** of agricultural commodities across Indian markets.

## 📁 Project Structure

```
crop-price-predictor/
├── app.py                    # Flask web application (main entry point)
├── train_model.py            # Model training script (generates .pkl files)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   └── Dataset.csv           # ⚠️ Place your dataset here
│
├── model/
│   ├── crop_price_model.pkl  # Trained model + encoder (auto-generated)
│   └── model_results.pkl     # Model comparison results (auto-generated)
│
├── static/
│   ├── css/
│   │   └── style.css         # Main stylesheet
│   ├── js/
│   │   └── main.js           # Frontend JavaScript
│   └── images/               # Image assets
│
└── templates/
    ├── index.html             # Home page (prediction form)
    └── about.html             # About page (project details)
```

## 🚀 Setup & Run

### 1. Install Python
Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/)

### 2. Create an Environment and Install Dependencies
```bash
# Windows:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux:
# python3 -m venv .venv
# .venv/bin/python -m pip install -r requirements.txt
```

### 3. Place Your Dataset
The training dataset is not included. Copy your `Dataset.csv` file into the `data/` folder. It must include `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`, and `Modal_Price` columns.

### 4. Train the Model (generates .pkl files)
```bash
python train_model.py
```

### 5. Run the Website
```bash
# Windows:
.\.venv\Scripts\python.exe app.py
# macOS/Linux:
# .venv/bin/python app.py
```

Open **http://127.0.0.1:5000** in your browser.

If the dataset or trained model is missing, the website starts in setup-required mode and disables predictions. It does not generate sample prices; add `data/Dataset.csv` and run the training step to enable predictions.

## 🧠 Models Evaluated
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The best performing model is automatically selected and saved.

## 🛠 Tech Stack
- **Backend:** Python, Flask
- **ML:** scikit-learn, XGBoost, LightGBM
- **Frontend:** HTML5, CSS3, JavaScript
- **Data:** Pandas, NumPy
