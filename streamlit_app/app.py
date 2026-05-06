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
    init_db,
    add_patient,
    get_patient,
    log_prediction,
    get_patient_history
)

# ---------------------------------------------------
# REPORT
# ---------------------------------------------------
try:
    from src.reports.medical_report import generate_medical_report
    REPORT_AVAILABLE = True
except Exception:
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

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict"
)

# ---------------------------------------------------
# INIT DATABASE
# ---------------------------------------------------
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_system():

    predictor = None
    gradcam = None

    try:

        predictor = Predictor(
            model_path=str(Config.get_model_path())
        )

        try:

            gradcam = GradCAMVisualizer(
                predictor.get_model()
            )

        except Exception as e:
            print("GradCAM disabled:", e)

    except Exception as e:
        st.error(f"Model loading failed: {e}")

    return predictor, gradcam


predictor, gradcam = load_system()

# ---------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------
st.sidebar.subheader("⚙️ System Status")

if predictor is not None:
    st.sidebar.success("Model Loaded")
else:
    st.sidebar.error("Model Failed")

if gradcam is not None:
    st.sidebar.success("Grad-CAM Ready")
else:
    st.sidebar.warning("Grad-CAM Disabled")

# ---------------------------------------------------
# API CALL
# ---------------------------------------------------
def api_predict(image, api_key):

    _, img_encoded = cv2.imencode(".jpg", image)

    files = {
        "file": (
            "image.jpg",
            img_encoded.tobytes(),
            "image/jpeg"
        )
    }

    headers = {
        "x-api-key": api_key
    }

    response = requests.post(
        API_URL,
        files=files,
        headers=headers,
        timeout=60
    )

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

# ---------------------------------------------------
# API SETTINGS
# ---------------------------------------------------
st.sidebar.subheader("🔐 API Settings")

api_mode = st.sidebar.checkbox(
    "Use API (SaaS Mode)"
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password"
)

# ---------------------------------------------------
# PATIENT INFO
# ---------------------------------------------------
st.sidebar.subheader("🧾 Patient Information")

if "patient_id" not in st.session_state:
    st.session_state.patient_id = (
        f"PAT-{uuid.uuid4().hex[:6].upper()}"
    )

patient_name = st.sidebar.text_input(
    "Patient Name",
    value="Shivam"
)

patient_id = st.sidebar.text_input(
    "Patient ID",
    value=st.session_state.patient_id
)

age = st.sidebar.number_input(
    "Age",
    0,
    120,
    30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "🏠 Home":

    st.title("🧠 Diabetic Retinopathy Screening AI")

    st.markdown("""
## AI-Powered Retinal Disease Detection System

### Features
- ✔ Diabetic Retinopathy Classification
- ✔ Grad-CAM Explainable AI
- ✔ Clinical PDF Report Generation
- ✔ Patient History Management
- ✔ FastAPI Backend Support
- ✔ Streamlit Interactive Dashboard

### Model
EfficientNetB3-based Deep Learning System

### Supported Classes
- No DR
- Mild NPDR
- Moderate NPDR
- Severe NPDR
- Proliferative DR
""")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Analyze Retinal Image")

    if predictor is None:
        st.error("Model unavailable")
        st.stop()

    file = st.file_uploader(
        "Upload Fundus Image",
        type=["png", "jpg", "jpeg"]
    )

    if file:

        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns([2, 1])

        with col1:

            st.image(
                image_np,
                caption="Uploaded Image",
                width="stretch"
            )

        # ---------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------
        with st.spinner("Analyzing retinal image..."):

            start_time = time.time()

            try:

                # API MODE
                if api_mode and api_key.strip():

                    res = api_predict(
                        image_np,
                        api_key
                    )

                    label = res["predicted_disease"]
                    conf = res["confidence"]

                    probs = list(
                        res["probabilities"].values()
                    )

                    st.success(
                        "✅ Prediction served from API"
                    )

                else:
                    raise Exception("Local inference")

            except Exception:

                label, conf, probs = predictor.predict(
                    image_np
                )

                st.info(
                    "💻 Prediction served locally"
                )

            end_time = time.time()

        # ---------------------------------------------------
        # DATABASE SAVE
        # ---------------------------------------------------
        try:

            if not get_patient(patient_id):

                add_patient(
                    patient_id,
                    patient_name,
                    age,
                    gender
                )

            log_prediction(
                patient_id,
                label,
                float(conf)
            )

        except Exception as e:
            st.warning(f"Database warning: {e}")

        # ---------------------------------------------------
        # UI OUTPUT
        # ---------------------------------------------------
        class_id = int(np.argmax(probs))

        risk_map = {
            0: "Low Risk 🟢",
            1: "Moderate Risk 🟡",
            2: "Moderate Risk 🟡",
            3: "High Risk 🔴",
            4: "High Risk 🔴"
        }

        with col2:

            st.markdown(f"## {label}")

            confidence_percent = conf * 100

            st.metric(
                "Confidence",
                f"{confidence_percent:.2f}%"
            )

            if confidence_percent >= 75:
                st.success("High Confidence")
            elif confidence_percent >= 50:
                st.warning("Moderate Confidence")
            else:
                st.error("Low Confidence")

            st.progress(float(conf))

            st.write(risk_map[class_id])

            st.caption(
                f"Inference time: "
                f"{end_time-start_time:.2f}s"
            )

        # ---------------------------------------------------
        # PATIENT PREVIEW
        # ---------------------------------------------------
        st.subheader("🧾 Patient Preview")

        st.write({
            "Name": patient_name,
            "ID": patient_id,
            "Age": age,
            "Gender": gender
        })

        # ---------------------------------------------------
        # PROBABILITIES
        # ---------------------------------------------------
        st.subheader("📊 Prediction Probabilities")

        df = pd.DataFrame({
            "Class": predictor.get_classes(),
            "Probability": probs
        })

        st.bar_chart(
            df.set_index("Class")
        )

        # ---------------------------------------------------
        # GRAD-CAM
        # ---------------------------------------------------
        st.subheader("🔥 AI Attention Map")

        heatmap = None

        if gradcam is not None:

            try:

                processed = predictor.preprocess(
                    image_np
                )

                processed = np.expand_dims(
                    processed,
                    axis=0
                )

                heatmap = gradcam.generate_cam(
                    processed
                )

                overlay = gradcam.overlay_heatmap(
                    image_np,
                    heatmap
                )

                st.image(
                    overlay,
                    caption="Grad-CAM Visualization",
                    width="stretch"
                )

            except Exception as e:

                st.warning(
                    f"Grad-CAM unavailable: {e}"
                )

        # ---------------------------------------------------
        # REPORT
        # ---------------------------------------------------
        st.subheader("📄 Medical Report")

        if REPORT_AVAILABLE:

            if st.button("Generate Report"):

                if not patient_name.strip():

                    st.warning(
                        "Please enter patient name"
                    )

                    st.stop()

                try:

                    with st.spinner(
                        "Generating PDF report..."
                    ):

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

                    st.success(
                        "✅ Medical report generated successfully"
                    )

                    with open(pdf_path, "rb") as pdf_file:

                        st.download_button(
                            label="📥 Download Medical Report",
                            data=pdf_file,
                            file_name=(
                                f"{patient_id}_medical_report.pdf"
                            ),
                            mime="application/pdf",
                            width="stretch"
                        )

                except Exception as e:

                    st.error(
                        f"Report generation failed: {e}"
                    )

        else:

            st.warning(
                "Medical report module unavailable"
            )

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------
elif page == "📜 History":

    st.title("📜 Patient History")

    search_id = st.text_input(
        "Enter Patient ID"
    )

    if search_id:

        try:

            history = get_patient_history(
                search_id
            )

            if not history:

                st.warning(
                    "No records found"
                )

            else:

                df = pd.DataFrame(history)

                st.dataframe(
                    df,
                    width="stretch"
                )

                st.download_button(
                    "📥 Download History",
                    df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=(
                        f"{search_id}_history.csv"
                    ),
                    mime="text/csv",
                    width="stretch"
                )

        except Exception as e:

            st.error(
                f"History error: {e}"
            )

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
else:

    st.title("ℹ️ About")

    st.markdown("""
## Diabetic Retinopathy Screening AI

AI-powered diabetic retinopathy detection system designed for early-stage retinal disease screening.

### Features
- EfficientNetB3-based classifier
- Explainable AI with Grad-CAM
- Clinical PDF report generation
- Patient history management
- FastAPI backend support
- Streamlit interactive frontend

### Tech Stack
- TensorFlow
- Streamlit
- FastAPI
- OpenCV
- SQLite
- ReportLab

### Developed For
Major Project Demonstration
""")