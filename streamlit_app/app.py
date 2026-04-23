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
from datetime import datetime
import requests
import cv2

from src.utils.config import Config
from src.inference.predictor import Predictor
from src.inference.grad_cam import GradCAMVisualizer

# SAFE IMPORT
try:
    from src.reports.medical_report import generate_medical_report
    REPORT_AVAILABLE = True
except:
    generate_medical_report = None
    REPORT_AVAILABLE = False

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="DR Screening AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD LOCAL MODEL
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
    url = "http://localhost:8000/predict"

    _, img_encoded = cv2.imencode(".jpg", image)

    files = {
        "file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    headers = {
        "x-api-key": api_key
    }

    response = requests.post(url, files=files, headers=headers)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()

# ---------------------------------------------------
# STATE
# ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_file" not in st.session_state:
    st.session_state.last_file = None

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
# HOME
# ---------------------------------------------------
if page == "🏠 Home":

    st.title("🧠 Diabetic Retinopathy Screening AI")

    st.markdown("""
### Detect severity of diabetic retinopathy

✔ Local + SaaS prediction  
✔ Confidence score  
✔ Grad-CAM visualization  
✔ Medical report export  
✔ History tracking  
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
            st.image(image_np, caption="Uploaded Image", width="stretch")

        # -------------------------
        # PREDICTION
        # -------------------------
        with st.spinner("🧠 Analyzing..."):
            try:
                if api_mode and api_key:
                    res = api_predict(image_np, api_key)

                    label = res["predicted_disease"]
                    conf = res["confidence"]
                    probs = list(res["probabilities"].values())
                else:
                    label, conf, probs = predictor.predict(image_np)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.stop()

        class_id = int(np.argmax(probs))

        risk_map = {
            0: "Low Risk 🟢",
            1: "Moderate Risk 🟡",
            2: "Moderate Risk 🟡",
            3: "High Risk 🔴",
            4: "High Risk 🔴"
        }
        risk = risk_map[class_id]

        with col2:
            st.markdown(
                f"<div style='padding:15px;background:#111;border-radius:10px;color:white'>{label}</div>",
                unsafe_allow_html=True
            )

            st.metric("Confidence", f"{conf*100:.2f}%")
            st.progress(float(conf))
            st.write(f"### {risk}")

        # ---------------------------------------------------
        # PROBABILITY
        # ---------------------------------------------------
        st.subheader("📊 Probability Distribution")

        df = pd.DataFrame({
            "Class": predictor.get_classes(),
            "Probability": probs
        })

        st.bar_chart(df.set_index("Class"))

        # ---------------------------------------------------
        # GRAD-CAM (LOCAL ONLY)
        # ---------------------------------------------------
        st.subheader("🔥 AI Attention Map")

        heatmap = None

        if not api_mode and gradcam is not None:
            try:
                processed = predictor.preprocess(image_np)
                processed = np.expand_dims(processed, axis=0)

                heatmap = gradcam.generate_cam(processed)
                overlay = gradcam.overlay_heatmap(image_np, heatmap)

                c1, c2 = st.columns(2)

                with c1:
                    st.image(image_np, caption="Original", width="stretch")

                with c2:
                    st.image(overlay, caption="Grad-CAM", width="stretch")

            except Exception as e:
                st.error(f"Grad-CAM error: {str(e)}")
        else:
            st.info("Grad-CAM available only in local mode")

        st.info("⚠️ AI screening tool. Not a medical diagnosis.")

        # ---------------------------------------------------
        # REPORT
        # ---------------------------------------------------
        st.subheader("📄 Medical Report")

        if not REPORT_AVAILABLE:
            st.warning("Install 'reportlab' to enable report download")
        else:
            if st.button("Generate Hospital Report"):
                with st.spinner("Generating report..."):
                    try:
                        pdf_path = generate_medical_report(
                            image=image_np,
                            prediction=label,
                            confidence=conf,
                            probabilities=probs,
                            class_names=predictor.get_classes(),
                            heatmap=heatmap
                        )

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="⬇ Download Report",
                                data=f,
                                file_name="DR_Report.pdf",
                                mime="application/pdf"
                            )

                    except Exception as e:
                        st.error(f"Report generation failed: {str(e)}")

        # ---------------------------------------------------
        # HISTORY
        # ---------------------------------------------------
        if st.session_state.last_file != file.name:
            st.session_state.history.append({
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Image": file.name,
                "Prediction": label,
                "Confidence": f"{conf*100:.2f}%"
            })
            st.session_state.last_file = file.name

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------
elif page == "📜 History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width="stretch")

        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False),
            file_name="history.csv",
            mime="text/csv"
        )

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
else:

    st.title("ℹ️ About Project")

    st.markdown("""
### Diabetic Retinopathy Classification

- Dataset: APTOS 2019  
- Model: EfficientNetB3  
- Framework: TensorFlow + Streamlit + FastAPI  

Hybrid Local + SaaS AI system.
""")