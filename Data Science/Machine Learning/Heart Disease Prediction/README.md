# ❤️ Heart Disease Prediction & Flask Web App

An end-to-end machine learning project to predict heart disease risk using a K-Nearest Neighbors (KNN) classifier and serve predictions through a Flask web application.

## 🚀 Features
- **Model Training**: KNN classification trained on clinical heart disease features.
- **Model Serialization**: Saved model artifacts (`KNN_heart.pkl`, `scaler.pkl`, `columns.pkl`) via `joblib`.
- **Web Interface**: Interactive Flask web application (`app.py`) for real-time user input & risk prediction.

## 🛠 Setup & Running
```bash
pip install flask scikit-learn joblib numpy pandas
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.
