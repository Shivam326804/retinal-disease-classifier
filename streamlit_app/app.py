# ---------------------------------------------------
# SYSTEM
# ---------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
import cv2
import uuid
import time

from src.utils.config import Config
from src.inference.predictor import Predictor
from src.inference.grad_cam import GradCAMVisualizer

from src.api.database import (
    add_patient,
    get_patient,
    log_prediction,
    get_patient_history
)

# REPORT
try:
    from src.reports.medical_report import generate_medical_report
    REPORT_AVAILABLE = True
except:
    generate_medical_report = None
    REPORT_AVAILABLE = False

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="DR Screening AI",
    page_icon="🧠",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_system():
    predictor = Predictor(model_path=str(Config.get_model_path()))

    gradcam = None
    try:
        gradcam = GradCAMVisualizer(predictor.get_model())
    except Exception as e:
        print("GradCAM disabled:", e)

    return predictor, gradcam


predictor, gradcam = load_system()

# ---------------------------------------------------
# API CALL
# ---------------------------------------------------
def api_predict(image, api_key):
    _, img_encoded = cv2.imencode(".jpg", image)

    files = {"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
    headers = {"x-api-key": api_key}

    response = requests.post(API_URL, files=files, headers=headers)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Prediction", "📜 History", "ℹ️ About"]
)

st.sidebar.subheader("🔐 API Settings")
api_mode = st.sidebar.checkbox("Use API (SaaS Mode)")
api_key = st.sidebar.text_input("API Key", type="password")

# ---------------------------------------------------
# 🧾 PATIENT INFO (SESSION SAFE)
# ---------------------------------------------------
st.sidebar.subheader("🧾 Patient Information")

if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"

patient_name = st.sidebar.text_input("Patient Name")

patient_id = st.sidebar.text_input(
    "Patient ID",
    value=st.session_state.patient_id
)

age = st.sidebar.number_input("Age", 0, 120, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "🏠 Home":
    st.title("🧠 Diabetic Retinopathy Screening AI")
    st.markdown("""
✔ Local + SaaS prediction  
✔ Grad-CAM visualization  
✔ Clinical PDF reports  
✔ Patient tracking system  
""")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Analyze Retinal Image")

    file = st.file_uploader("Upload Fundus Image", type=["png","jpg","jpeg"])

    if file:

        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns([2,1])

        with col1:
            st.image(image_np, caption="Uploaded Image", use_container_width=True)

        # -------------------------
        # PREDICTION (SAFE + FALLBACK)
        # -------------------------
        with st.spinner("Analyzing..."):

            start_time = time.time()

            try:
                if api_mode and api_key:
                    res = api_predict(image_np, api_key)
                    label = res["predicted_disease"]
                    conf = res["confidence"]
                    probs = list(res["probabilities"].values())
                else:
                    raise Exception("Using local")

            except:
                label, conf, probs = predictor.predict(image_np)

            end_time = time.time()

        # -------------------------
        # DB SAVE (SAFE)
        # -------------------------
        if not get_patient(patient_id):
            add_patient(patient_id, patient_name, age, gender)

        log_prediction(patient_id, label, float(conf))

        # -------------------------
        # UI OUTPUT
        # -------------------------
        class_id = int(np.argmax(probs))

        risk_map = {
            0: "Low Risk 🟢",
            1: "Moderate Risk 🟡",
            2: "Moderate Risk 🟡",
            3: "High Risk 🔴",
            4: "High Risk 🔴"
        }

        with col2:
            st.markdown(f"### {label}")
            st.metric("Confidence", f"{conf*100:.2f}%")
            st.progress(float(conf))
            st.write(risk_map[class_id])
            st.caption(f"Inference time: {end_time-start_time:.2f}s")

        # -------------------------
        # PATIENT PREVIEW
        # -------------------------
        st.subheader("🧾 Patient Preview")
        st.write({
            "Name": patient_name,
            "ID": patient_id,
            "Age": age,
            "Gender": gender
        })

        # -------------------------
        # PROBABILITY
        # -------------------------
        df = pd.DataFrame({
            "Class": predictor.get_classes(),
            "Probability": probs
        })
        st.bar_chart(df.set_index("Class"))

        # -------------------------
        # GRAD-CAM (SAFE)
        # -------------------------
        st.subheader("🔥 AI Attention Map")

        heatmap = None

        if gradcam is not None:
            try:
                processed = predictor.preprocess(image_np)
                processed = np.expand_dims(processed, axis=0)

                heatmap = gradcam.generate_cam(processed)
                overlay = gradcam.overlay_heatmap(image_np, heatmap)

                st.image(overlay, caption="Grad-CAM", use_container_width=True)

            except:
                st.warning("Grad-CAM unavailable")

        # -------------------------
        # REPORT
        # -------------------------
        st.subheader("📄 Medical Report")

        if REPORT_AVAILABLE:
            if st.button("Generate Report"):

                if not patient_name:
                    st.warning("Enter patient name")
                    st.stop()

                pdf_path = generate_medical_report(
                    image=image_np,
                    prediction=label,
                    confidence=conf,
                    probabilities=probs,
                    class_names=predictor.get_classes(),
                    heatmap=heatmap,
                    patient_data={
                        "name": patient_name,
                        "id": patient_id,
                        "age": age,
                        "gender": gender
                    }
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Report",
                        f,
                        file_name=f"{patient_id}_report.pdf"
                    )

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------
elif page == "📜 History":

    st.title("📜 Patient History")

    search_id = st.text_input("Enter Patient ID")

    if search_id:
        history = get_patient_history(search_id)

        if not history:
            st.warning("No records found")
        else:
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download History",
                df.to_csv(index=False),
                file_name=f"{search_id}_history.csv"
            )

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
else:
    st.title("ℹ️ About")
    st.markdown("""
AI-powered diabetic retinopathy detection system  
Built with TensorFlow, Streamlit, and FastAPI  
Includes explainability and clinical reporting  
""")