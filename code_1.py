import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import os

@st.cache_resource
def load_models():
    return (
        load_model("cnn_model.h5"),
        joblib.load("rf_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoders.pkl"),
        joblib.load("X_columns.pkl")
    )

cnn_model, rf_model, scaler, label_encoders, X_columns = load_models()

st.title("Brain Tumor Risk Predictor")
if st.button("Predict"):
    st.success(f"This patient has a {risk_percent}% chance of developing a tumor.")
