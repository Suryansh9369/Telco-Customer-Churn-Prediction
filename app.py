import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load saved objects
# -----------------------------
model = joblib.load("model.pkl")
encoded_columns = joblib.load("encoded_columns.pkl")
churn_encoder = joblib.load("churn_label_encoder.pkl")

st.set_page_config(page_title="Telco Churn Prediction")
st.title("📊 Telco Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)

tenure_group = st.selectbox(
    "Tenure Group",
    ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    # Create raw input dataframe
    input_data = pd.DataFrame([{
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless_billing,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_group": tenure_group
    }])

    # One-hot encode using get_dummies
    input_encoded = pd.get_dummies(input_data)

    # Align columns with training data
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Predict
    churn_pred = model.predict(input_encoded)[0]
    churn_prob = model.predict_proba(input_encoded)[0][1]

    churn_label = churn_encoder.inverse_transform([churn_pred])[0]

    # Output
    st.subheader(f"Prediction: **{churn_label}**")
    st.write(f"Churn Probability: **{churn_prob:.2%}**")
    
