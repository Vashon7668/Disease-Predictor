from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  

model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.joblib")
df = pd.read_csv(r"C:\Users\Harikar\LSM_Project\cleaned_balanced_dataset.csv")

token_counter = 0
average_consultation_time = 10

@app.route("/predict", methods=["POST"])
def predict():
    global token_counter
    data = request.json

    try:
        features = [
            int(data["fever"]),
            int(data["cough"]),
            int(data["fatigue"]),
            int(data["breathing"]),
            int(data["age"]),
            int(data["gender"]),
            int(data["blood_pressure"]),
            int(data["cholesterol"]),
            1  # dummy or default value
        ]

        if len(features) != 9:
            return jsonify({"error": "Expected 9 features"}), 400

        disease_label = model.predict([features])[0]
        disease_name = label_encoder.inverse_transform([disease_label])[0]
        matching_rows = df[df["Disease"] == disease_label]

        if matching_rows.empty:
            return jsonify({"error": "Disease not found in dataset"}), 404

        outcome = matching_rows["Outcome Variable"].values[0]
        advice = f"{disease_name} ({'Consult a doctor' if outcome == 1 else 'No immediate consultation needed'})"

        token_counter += 1
        estimated_wait_time = token_counter * average_consultation_time

        return jsonify({
            "result": advice,
            "token": f"#{token_counter}",
            "wait_time": f"~{estimated_wait_time} minutes"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Disease Prediction API"})

if __name__ == "__main__":
    app.run(debug=False, port=5001)
