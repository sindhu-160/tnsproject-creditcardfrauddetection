import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load trained model and features
model = joblib.load("fraud_model.joblib")
feature_names = joblib.load("feature_names.joblib")

st.title("💳 Credit Card Fraud Detection Application")

# Sidebar for user inputs
st.sidebar.header("Transaction Details")

# Core features
time = st.sidebar.number_input("Transaction Time (Seconds)", min_value=0.0, value=100.0)
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)

# PCA Features V1 - V28 in an expander
with st.sidebar.expander("V1 to V28 Principal Component Features"):
    pca_inputs = {}
    for i in range(1, 29):
        col_name = f"V{i}"
        pca_inputs[col_name] = st.number_input(f"Enter {col_name}", value=0.0)

# Combine inputs into a dictionary
input_dict = {
    'Time': time,
    **pca_inputs,
    'Amount': amount
}

# Construct DataFrame matching model features
input_df = pd.DataFrame([input_dict])[feature_names]

# Display input summary
st.subheader("Transaction Input Summary")
st.dataframe(input_df, use_container_width=True)

# Predict on button click
if st.button("Predict Fraud Status", type="primary"):
    prediction = model.predict(input_df)[0]
    
    # Calculate probability score
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("⚠️ High Risk: Potential Fraudulent Transaction Detected!")
        if probability is not None:
            st.write(f"Fraud Probability: **{probability * 100:.2f}%**")
    else:
        st.success("✅ Safe: Low Risk / Normal Transaction")
        if probability is not None:
            st.write(f"Fraud Probability: **{probability * 100:.2f}%**")