import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# === Load models and utilities ===
cnn_model = load_model("cnn_model.h5")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_columns = joblib.load("X_columns.pkl")

categorical_cols = ['Gender', 'Smoking', 'Alcohol', 'FamilyHistory', 'Occupation', 'Diet', 'ExerciseFreq']

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dim
    img = np.expand_dims(img, axis=0)   # batch dim
    return img

def is_mri_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img is not None and len(img.shape) == 2

def encode_and_scale_lifestyle(data_dict):
    df = pd.DataFrame([data_dict])
    df["BMI"] = df["WeightKg"] / ((df["HeightCm"] / 100) ** 2)
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])
    df = df[X_columns]
    return scaler.transform(df)

def predict_lifestyle_risk(data_dict):
    try:
        X = encode_and_scale_lifestyle(data_dict)
        prob = rf_model.predict_proba(X)[0][1]
        return prob
    except:
        return 0.0

def predict_mri_risk(image_path):
    img_input = preprocess_image(image_path)
    if img_input is not None:
        return cnn_model.predict(img_input, verbose=0).flatten()[0]
    return 0.0

def predict_combined_risk(data_dict, image_path):
    mri_risk = predict_mri_risk(image_path)
    lifestyle_risk = predict_lifestyle_risk(data_dict)
    return (mri_risk + lifestyle_risk) / 2

# === Streamlit App ===
st.set_page_config(page_title="Brain Tumor Risk Prediction", layout="centered")
st.title("🧠 Brain Tumor Risk Predictor")

st.markdown("""
Choose an input mode and fill in the required information to estimate tumor risk:
""")

option = st.radio("Choose input method", ["Lifestyle Data Only", "MRI Image Only", "Both"])

# ========== Lifestyle Inputs ==========
lifestyle_data = {}
image_path = None

if option != "MRI Image Only":
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["male", "female"])
    smoking = st.selectbox("Smoking", ["yes", "no"])
    alcohol = st.selectbox("Alcohol", ["yes", "no"])
    family_history = st.selectbox("Family History", ["yes", "no"])
    occupation = st.selectbox("Occupation", ["manual", "office"])
    diet = st.selectbox("Diet", ["good", "poor"])
    exercise_freq = st.selectbox("Exercise Frequency", ["low", "medium", "high"])
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=200, value=65)

    lifestyle_data = {
        "Age": age,
        "Gender": gender,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "FamilyHistory": family_history,
        "Occupation": occupation,
        "Diet": diet,
        "ExerciseFreq": exercise_freq,
        "HeightCm": height_cm,
        "WeightKg": weight_kg,
    }

if option != "Lifestyle Data Only":
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        save_path = os.path.join("temp_uploads", uploaded_file.name)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if not is_mri_image(save_path):
            st.warning("⚠️ Uploaded image is not a valid MRI scan.")
            st.stop()
        else:
            image_path = save_path

# ========== Predict ==========
if st.button("🔍 Predict Tumor Risk"):
    if option == "Lifestyle Data Only":
        risk = predict_lifestyle_risk(lifestyle_data)
    elif option == "MRI Image Only":
        if image_path:
            risk = predict_mri_risk(image_path)
        else:
            st.warning("Please upload a valid MRI image.")
            st.stop()
    else:
        if not lifestyle_data or not image_path:
            st.warning("Please fill lifestyle data and upload an MRI image.")
            st.stop()
        risk = predict_combined_risk(lifestyle_data, image_path)

    st.success(f"This patient has a {int(risk * 100)}% chance of developing a brain tumor.")
