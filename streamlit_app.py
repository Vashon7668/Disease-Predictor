import streamlit as st 
import joblib
import pandas as pd
import numpy as np
import random
import qrcode
from io import BytesIO
from PIL import Image
from fpdf import FPDF

# Load model and encoder
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.joblib")
df = pd.read_csv("cleaned_balanced_dataset.csv")

# Hospital and camp assignments
assignments = [
    {"hospital": "Apollo Hospitals", "camp": "Wellness Camp A", "address": "Road 1, Hyderabad"},
    {"hospital": "AIIMS Delhi", "camp": "Health Camp B", "address": "Ansari Nagar, New Delhi"},
    {"hospital": "Fortis Bangalore", "camp": "Free Checkup Camp C", "address": "Bannerghatta Road, Bangalore"},
    {"hospital": "Narayana Health", "camp": "Outreach Camp D", "address": "Hosur Road, Bangalore"},
    {"hospital": "Manipal Hospitals", "camp": "Urban Clinic Camp E", "address": "Old Airport Road, Bangalore"}
]

# Session-state token counter
if "token_number" not in st.session_state:
    st.session_state.token_number = 0  # Start from 0 so first token becomes #1

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

    # Prediction
    label = model.predict([features])[0]
    disease = label_encoder.inverse_transform([label])[0]
    match = df[df["Disease"] == label]

    if not match.empty:
        outcome = match["Outcome Variable"].values[0]
        advice = "Consult a doctor." if outcome == 1 else "No immediate consultation needed."

        # Assign hospital and camp
        assignment = random.choice(assignments)
        hospital = assignment["hospital"]
        camp = assignment["camp"]
        address = assignment["address"]

        # Token number and estimated wait time
        st.session_state.token_number += 1
        token_number = f"#{st.session_state.token_number}"
        average_consult_time = 10  # in minutes
        estimated_wait_time = st.session_state.token_number * average_consult_time

        # Display result
        st.markdown(f"""
        ### 🧾 Result Summary
        - **Disease:** {disease}  
        - **Advice:** {advice}  
        - **Hospital:** {hospital}  
        - **Camp:** {camp}  
        - **Address:** {address}  
        - **Token Number:** {token_number}  
        - **Estimated Wait Time:** ~{estimated_wait_time} minutes
        """)

        # QR code content
        qr_text = f"Disease: {disease}\nAdvice: {advice}\nToken: {token_number}\nHospital: {hospital}\nCamp: {camp}\nAddress: {address}"
        qr_img = qrcode.make(qr_text)

        # Show QR code
        qr_buf_display = BytesIO()
        qr_img.save(qr_buf_display, format="PNG")
        qr_buf_display.seek(0)
        st.image(qr_buf_display, caption="Scan QR Code for Details", use_container_width=True)

        # QR download
        qr_buf_download = BytesIO()
        qr_img.save(qr_buf_download, format="PNG")
        st.download_button("📥 Download QR Code", data=qr_buf_download.getvalue(), file_name="qr_code.png", mime="image/png")

        # PDF generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=qr_text)
        pdf_output = pdf.output(dest="S").encode("latin1")
        st.download_button("📄 Download Report PDF", data=pdf_output, file_name="prediction_report.pdf", mime="application/pdf")
    else:
        st.error("Disease not found in reference dataset.")
