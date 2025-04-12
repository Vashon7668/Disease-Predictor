import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.joblib")
df = pd.read_csv("cleaned_balanced_dataset.csv")

st.title("🩺 Disease Prediction App")

# Input form
with st.form("health_form"):
    st.subheader("Enter Patient Details")
    fever = st.radio("Fever", ["Yes", "No"])
    cough = st.radio("Cough", ["Yes", "No"])
    fatigue = st.radio("Fatigue", ["Yes", "No"])
    breathing = st.radio("Difficulty Breathing", ["Yes", "No"])
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Low"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Mapping inputs
    yesno_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}
    bp_map = {"Normal": 0, "High": 1, "Low": 2}
    chol_map = {"Normal": 0, "High": 1, "Low": 2}

    features = [
        yesno_map[fever],
        yesno_map[cough],
        yesno_map[fatigue],
        yesno_map[breathing],
        age,
        gender_map[gender],
        bp_map[bp],
        chol_map[cholesterol],
        1  # Placeholder for missing feature
    ]

    # Make prediction
    label = model.predict([features])[0]
    disease = label_encoder.inverse_transform([label])[0]

    # Determine consultation advice
    match = df[df["Disease"] == label]
    if not match.empty:
        outcome = match["Outcome Variable"].values[0]
        advice = f"🧾 Prediction: {disease} — "
        advice += "Consult a doctor." if outcome == 1 else "No immediate consultation needed."
        st.success(advice)
    else:
        st.error("Disease not found in reference dataset.")
