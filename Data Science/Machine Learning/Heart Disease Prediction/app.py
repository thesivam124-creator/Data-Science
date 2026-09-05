"""
Heart Disease Prediction - Flask Web App
==========================================
This app serves a simple web frontend for the KNN model trained in
HeartdiseaseFinal.ipynb. It expects the following files (produced by the
notebook's final cell) to sit in the SAME folder as this script:

    KNN_heart.pkl   -> joblib.dump(models['KNN'], 'KNN_heart.pkl')
    scaler.pkl      -> joblib.dump(scaler, 'scaler.pkl')
    columns.pkl     -> joblib.dump(X.columns.tolist(), 'columns.pkl')

Run with:
    pip install flask scikit-learn joblib numpy pandas
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "KNN_heart.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")

model = None
scaler = None
model_columns = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as exc:  # noqa: BLE001
    load_error = str(exc)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML template (kept in this single file for simplicity)
# ---------------------------------------------------------------------------
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heart Disease Risk Predictor</title>
<style>
    :root {
        --accent: #d64550;
        --accent-dark: #b5323c;
        --bg: #f5f6fa;
        --card-bg: #ffffff;
        --text: #2b2d3e;
        --muted: #6b6f80;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        display: flex;
        justify-content: center;
        padding: 40px 16px;
    }
    .card {
        background: var(--card-bg);
        max-width: 720px;
        width: 100%;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        padding: 32px 36px 40px;
    }
    h1 {
        text-align: center;
        font-size: 1.6rem;
        margin-bottom: 4px;
        color: var(--accent-dark);
    }
    .subtitle {
        text-align: center;
        color: var(--muted);
        margin-bottom: 28px;
        font-size: 0.95rem;
    }
    form {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px 20px;
    }
    .field { display: flex; flex-direction: column; }
    .field.full { grid-column: 1 / -1; }
    label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 6px;
        color: var(--text);
    }
    input, select {
        padding: 10px 12px;
        border: 1px solid #d9dbe3;
        border-radius: 8px;
        font-size: 0.95rem;
        background: #fbfbfd;
        color: var(--text);
    }
    input:focus, select:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(214,69,80,0.15);
    }
    button {
        grid-column: 1 / -1;
        margin-top: 8px;
        padding: 14px;
        border: none;
        border-radius: 10px;
        background: var(--accent);
        color: #fff;
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    button:hover { background: var(--accent-dark); }
    .result {
        margin-top: 26px;
        padding: 18px 20px;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        font-size: 1.05rem;
    }
    .result.high {
        background: #fde8e9;
        color: var(--accent-dark);
        border: 1px solid #f3c1c5;
    }
    .result.low {
        background: #e7f7ee;
        color: #1f7a4d;
        border: 1px solid #bfe8d1;
    }
    .prob {
        display: block;
        margin-top: 6px;
        font-size: 0.85rem;
        font-weight: 400;
        color: var(--muted);
    }
    .error {
        margin-top: 20px;
        padding: 14px 16px;
        border-radius: 10px;
        background: #fff4e5;
        color: #8a5a00;
        border: 1px solid #ffdca8;
        font-size: 0.9rem;
    }
    .footnote {
        margin-top: 22px;
        text-align: center;
        font-size: 0.75rem;
        color: var(--muted);
    }
</style>
</head>
<body>
<div class="card">
    <h1>&#10084;&#65039; Heart Disease Risk Predictor</h1>
    <p class="subtitle">Enter the patient's clinical details below to estimate heart disease risk (KNN model).</p>

    {% if load_error %}
    <div class="error">
        Model files could not be loaded: {{ load_error }}<br>
        Make sure KNN_heart.pkl, scaler.pkl and columns.pkl are in the same folder as app.py
        (generate them by running the last cell of HeartdiseaseFinal.ipynb).
    </div>
    {% endif %}

    <form method="POST">
        <div class="field">
            <label for="Age">Age</label>
            <input type="number" id="Age" name="Age" min="1" max="120" required value="{{ form.Age or 45 }}">
        </div>

        <div class="field">
            <label for="Sex">Sex</label>
            <select id="Sex" name="Sex">
                <option value="M" {{ 'selected' if form.Sex=='M' else '' }}>Male</option>
                <option value="F" {{ 'selected' if form.Sex=='F' else '' }}>Female</option>
            </select>
        </div>

        <div class="field">
            <label for="ChestPainType">Chest Pain Type</label>
            <select id="ChestPainType" name="ChestPainType">
                <option value="ATA" {{ 'selected' if form.ChestPainType=='ATA' else '' }}>Atypical Angina (ATA)</option>
                <option value="NAP" {{ 'selected' if form.ChestPainType=='NAP' else '' }}>Non-Anginal Pain (NAP)</option>
                <option value="ASY" {{ 'selected' if form.ChestPainType=='ASY' else '' }}>Asymptomatic (ASY)</option>
                <option value="TA" {{ 'selected' if form.ChestPainType=='TA' else '' }}>Typical Angina (TA)</option>
            </select>
        </div>

        <div class="field">
            <label for="RestingBP">Resting Blood Pressure (mm Hg)</label>
            <input type="number" id="RestingBP" name="RestingBP" min="0" max="300" required value="{{ form.RestingBP or 120 }}">
        </div>

        <div class="field">
            <label for="Cholesterol">Cholesterol (mg/dl)</label>
            <input type="number" id="Cholesterol" name="Cholesterol" min="0" max="700" required value="{{ form.Cholesterol or 200 }}">
        </div>

        <div class="field">
            <label for="FastingBS">Fasting Blood Sugar &gt; 120 mg/dl</label>
            <select id="FastingBS" name="FastingBS">
                <option value="0" {{ 'selected' if form.FastingBS=='0' else '' }}>No</option>
                <option value="1" {{ 'selected' if form.FastingBS=='1' else '' }}>Yes</option>
            </select>
        </div>

        <div class="field">
            <label for="RestingECG">Resting ECG</label>
            <select id="RestingECG" name="RestingECG">
                <option value="Normal" {{ 'selected' if form.RestingECG=='Normal' else '' }}>Normal</option>
                <option value="ST" {{ 'selected' if form.RestingECG=='ST' else '' }}>ST-T abnormality (ST)</option>
                <option value="LVH" {{ 'selected' if form.RestingECG=='LVH' else '' }}>Left Ventricular Hypertrophy (LVH)</option>
            </select>
        </div>

        <div class="field">
            <label for="MaxHR">Max Heart Rate Achieved</label>
            <input type="number" id="MaxHR" name="MaxHR" min="40" max="250" required value="{{ form.MaxHR or 150 }}">
        </div>

        <div class="field">
            <label for="ExerciseAngina">Exercise-Induced Angina</label>
            <select id="ExerciseAngina" name="ExerciseAngina">
                <option value="N" {{ 'selected' if form.ExerciseAngina=='N' else '' }}>No</option>
                <option value="Y" {{ 'selected' if form.ExerciseAngina=='Y' else '' }}>Yes</option>
            </select>
        </div>

        <div class="field">
            <label for="Oldpeak">Oldpeak (ST depression)</label>
            <input type="number" step="0.1" id="Oldpeak" name="Oldpeak" min="-5" max="10" required value="{{ form.Oldpeak or 1.0 }}">
        </div>

        <div class="field">
            <label for="ST_Slope">ST Slope</label>
            <select id="ST_Slope" name="ST_Slope">
                <option value="Up" {{ 'selected' if form.ST_Slope=='Up' else '' }}>Up</option>
                <option value="Flat" {{ 'selected' if form.ST_Slope=='Flat' else '' }}>Flat</option>
                <option value="Down" {{ 'selected' if form.ST_Slope=='Down' else '' }}>Down</option>
            </select>
        </div>

        <button type="submit" {{ 'disabled' if load_error else '' }}>Predict Risk</button>
    </form>

    {% if prediction is not none %}
        <div class="result {{ 'high' if prediction == 1 else 'low' }}">
            {% if prediction == 1 %}
                &#9888;&#65039; High risk of heart disease
            {% else %}
                &#9989; Low risk of heart disease
            {% endif %}
            {% if probability is not none %}
                <span class="prob">Model confidence: {{ probability }}%</span>
            {% endif %}
        </div>
    {% endif %}

    <p class="footnote">This tool is for educational purposes only and is not a substitute for professional medical advice.</p>
</div>
</body>
</html>
"""


def build_feature_row(form):
    """Convert raw form input into the one-hot-encoded row the model expects,
    matching the exact column order saved in columns.pkl (pd.get_dummies with
    drop_first=True on the original heart.csv columns)."""

    age = float(form["Age"])
    resting_bp = float(form["RestingBP"])
    cholesterol = float(form["Cholesterol"])
    fasting_bs = int(form["FastingBS"])
    max_hr = float(form["MaxHR"])
    oldpeak = float(form["Oldpeak"])

    sex = form["Sex"]
    chest_pain = form["ChestPainType"]
    resting_ecg = form["RestingECG"]
    exercise_angina = form["ExerciseAngina"]
    st_slope = form["ST_Slope"]

    # Start every one-hot column at 0, then flip the relevant ones to 1.
    row = {col: 0 for col in model_columns}

    row["Age"] = age
    row["RestingBP"] = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"] = fasting_bs
    row["MaxHR"] = max_hr
    row["Oldpeak"] = oldpeak

    if "Sex_M" in row and sex == "M":
        row["Sex_M"] = 1

    cp_col = f"ChestPainType_{chest_pain}"
    if cp_col in row:
        row[cp_col] = 1

    ecg_col = f"RestingECG_{resting_ecg}"
    if ecg_col in row:
        row[ecg_col] = 1

    if "ExerciseAngina_Y" in row and exercise_angina == "Y":
        row["ExerciseAngina_Y"] = 1

    slope_col = f"ST_Slope_{st_slope}"
    if slope_col in row:
        row[slope_col] = 1

    # Preserve exact column order expected by the scaler/model.
    ordered_values = [row[col] for col in model_columns]
    return pd.DataFrame([ordered_values], columns=model_columns)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()

        if load_error:
            return render_template_string(
                PAGE_TEMPLATE,
                load_error=load_error,
                prediction=None,
                probability=None,
                form=form_values,
            )

        try:
            input_df = build_feature_row(request.form)
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prediction = int(pred)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                probability = round(float(np.max(proba)) * 100, 1)
        except Exception as exc:  # noqa: BLE001
            return render_template_string(
                PAGE_TEMPLATE,
                load_error=f"Prediction failed: {exc}",
                prediction=None,
                probability=None,
                form=form_values,
            )

    return render_template_string(
        PAGE_TEMPLATE,
        load_error=load_error,
        prediction=prediction,
        probability=probability,
        form=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
