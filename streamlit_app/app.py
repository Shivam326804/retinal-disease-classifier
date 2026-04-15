"""
Streamlit Web Application - FINAL FIXED VERSION
"""

# ---------------------------------------------------
# SYSTEM FIXES
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
from typing import Optional, Tuple

from src.utils.config import Config
from src.inference.predictor import Predictor
from src.inference.grad_cam import GradCAMVisualizer

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Retinal Disease Classifier",
    page_icon="👁️",
    layout="wide"
)

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0b1220; }
h1, h2, h3 { color: white; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model() -> Tuple[Optional[Predictor], Optional[GradCAMVisualizer]]:
    model_path = Config.get_model_path()

    if model_path is None:
        return None, None

    predictor = Predictor(
        model_path=str(model_path),
        class_names=Config.DISEASE_CLASSES
    )

    gradcam = GradCAMVisualizer(predictor.model)

    return predictor, gradcam


predictor, gradcam = load_model()

# ---------------------------------------------------
# SAFETY CHECK (CRITICAL FIX)
# ---------------------------------------------------
if predictor is None or gradcam is None:
    st.error("❌ No trained model found. Train model first.")
    st.stop()

# 👇 Tell Pylance these are NOT None anymore
assert predictor is not None
assert gradcam is not None

# ---------------------------------------------------
# SESSION
# ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# NAVIGATION
# ---------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Confusion Matrix", "History", "About"]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "Home":
    st.title("👁️ Retinal Disease AI Classifier")

    st.markdown("""
Deep learning model to detect diabetic retinopathy.

### Classes:
- No DR
- Mild NPDR
- Moderate NPDR
- Severe NPDR
- Proliferative DR
""")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif page == "Prediction":

    st.title("🔍 Analyze Retinal Image")

    uploaded_file = st.file_uploader(
        "Upload Fundus Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")

        with st.spinner("🔄 Analyzing..."):

            try:
                pred, conf, probs = predictor.predict(image_np)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # RESULT
        with col2:
            st.success(pred)
            st.metric("Confidence", f"{conf * 100:.2f}%")

        st.divider()

        # TABLE
        st.subheader("📊 Prediction Breakdown")

        df = pd.DataFrame({
            "Disease": list(Config.DISEASE_CLASSES.values()),
            "Probability (%)": (probs * 100).round(2)
        }).sort_values(by="Probability (%)", ascending=False)

        st.dataframe(df, width="stretch")

        # CHART
        st.subheader("📈 Confidence Distribution")
        st.bar_chart(df.set_index("Disease"))

        # ---------------------------------------------------
        # GRAD-CAM (FIXED SAFE CHECK)
        # ---------------------------------------------------
        st.subheader("🔥 Model Attention (Grad-CAM)")

        try:
            processed_img = predictor.preprocess(image_np)

            heatmap = gradcam.generate_cam(
                processed_img,
                class_idx=int(np.argmax(probs))
            )

            overlay = gradcam.overlay_heatmap(image_np, heatmap)

            c1, c2 = st.columns(2)

            with c1:
                st.image(image, caption="Original", width="stretch")

            with c2:
                st.image(overlay, caption="Grad-CAM", width="stretch")

        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

        # INFO
        st.info(f"""
Prediction: **{pred}**  
Confidence: **{conf*100:.2f}%**

⚠️ Not a medical diagnosis.
""")

        # SAVE HISTORY
        st.session_state.history.append({
            "image": uploaded_file.name,
            "prediction": pred,
            "confidence": f"{conf*100:.2f}%"
        })

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
elif page == "Confusion Matrix":

    st.title("📊 Model Evaluation")

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report

        images = np.load(Config.PROCESSED_IMAGES)
        labels = np.load(Config.PROCESSED_LABELS)

        st.info(f"Loaded {len(images)} samples")

        preds = predictor.predict_batch(images)
        pred_labels = np.argmax(preds, axis=1)

        cm = confusion_matrix(labels, pred_labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=Config.CLASS_NAMES,
            yticklabels=Config.CLASS_NAMES
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        report = classification_report(
            labels,
            pred_labels,
            target_names=Config.CLASS_NAMES,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------
elif page == "History":

    st.title("📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch")

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
elif page == "About":

    st.title("ℹ️ About")

    st.write("""
Model: EfficientNet  
Task: Diabetic Retinopathy Classification  

This system analyzes retinal images using deep learning.
""")