from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import random

app = Flask(__name__)
CORS(app)

# Load model and data
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.joblib")
df = pd.read_csv("cleaned_balanced_dataset.csv")

# Token system
token_counter = 0
average_consultation_time = 10  # minutes

# Hospital + Camp assignments
assignments = [
    {"hospital": "Apollo Hospitals", "camp": "Wellness Camp A", "address": "Road 1, Hyderabad"},
    {"hospital": "AIIMS Delhi", "camp": "Health Camp B", "address": "Ansari Nagar, New Delhi"},
    {"hospital": "Fortis Bangalore", "camp": "Free Checkup Camp C", "address": "Bannerghatta Road, Bangalore"},
    {"hospital": "Narayana Health", "camp": "Outreach Camp D", "address": "Hosur Road, Bangalore"},
    {"hospital": "Manipal Hospitals", "camp": "Urban Clinic Camp E", "address": "Old Airport Road, Bangalore"}
]

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
            1  # Dummy placeholder value
        ]

        if len(features) != 9:
            return jsonify({"error": "Expected 9 features"}), 400

        # Predict
        disease_label = model.predict([features])[0]
        disease_name = label_encoder.inverse_transform([disease_label])[0]
        matching_rows = df[df["Disease"] == disease_label]

        if matching_rows.empty:
            return jsonify({"error": "Disease not found in dataset"}), 404

        # Outcome
        outcome = matching_rows["Outcome Variable"].values[0]
        advice = "Consult a doctor" if outcome == 1 else "No immediate consultation needed"

        # Assign hospital/camp/address
        assignment = random.choice(assignments)
        hospital = assignment["hospital"]
        camp = assignment["camp"]
        address = assignment["address"]

        # Token & wait time
        token_counter += 1
        token_number = f"#{token_counter}"
        wait_time = token_counter * average_consultation_time

        # Return full result
        return jsonify({
            "result": f"{disease_name} ({advice})",
            "disease": disease_name,
            "advice": advice,
            "token": token_number,
            "wait_time": f"~{wait_time} minutes",
            "hospital": hospital,
            "camp": camp,
            "address": address
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Disease Prediction API"})

if __name__ == "__main__":
    app.run(debug=False, port=5001)
