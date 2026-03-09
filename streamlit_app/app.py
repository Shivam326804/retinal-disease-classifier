"""
Streamlit Web Application
Interactive UI for retinal disease classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys
import cv2

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.inference import Predictor, GradCAMVisualizer

logger = setup_logger(__name__)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Retinal Disease Classifier",
    page_icon="👁️",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown("""
<style>
.main-header {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#2E86AB;
}
.sub-header{
    text-align:center;
    color:#666;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "predictor" not in st.session_state:
    st.session_state.predictor = None

if "gradcam" not in st.session_state:
    st.session_state.gradcam = None

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model_path = Path(Config.MODELS_DIR) / Config.MODEL_NAME

    if not model_path.exists():
        st.warning("Model not found. Train model first.")
        return None

    try:
        predictor = Predictor(str(model_path), Config.DISEASE_CLASSES)
        return predictor

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    st.markdown('<p class="main-header">Retinal Disease AI Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Based Diabetic Retinopathy Detection</p>', unsafe_allow_html=True)

    st.session_state.predictor = load_model()

    with st.sidebar:

        st.title("Navigation")

        page = st.radio(
            "Select Page",
            [
                "Home",
                "Make Prediction",
                "Model Information",
                "Prediction History",
                "About"
            ]
        )

    if page == "Home":
        show_home()

    elif page == "Make Prediction":
        show_prediction()

    elif page == "Model Information":
        show_model_info()

    elif page == "Prediction History":
        show_history()

    elif page == "About":
        show_about()


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

def show_home():

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("System Overview")

        st.write("""
This AI system automatically detects **diabetic retinopathy severity**
from retinal fundus images using deep learning models.
""")

    with col2:

        st.subheader("Disease Classes")

        for k, v in Config.DISEASE_CLASSES.items():
            st.write(f"{k} : {v}")


# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

def show_prediction():

    st.subheader("Retinal Disease Detection")

    if st.session_state.predictor is None:
        st.error("Model not loaded")
        return

    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)
    image_array = preprocessor.preprocess_image_array(np.array(image))

    predicted_class, confidence, probabilities = (
        st.session_state.predictor.predict(image_array)
    )

    with col2:

        st.subheader("Prediction")

        st.metric("Disease", predicted_class)
        st.metric("Confidence", f"{confidence:.2%}")

        prob_dist = (
            st.session_state.predictor
            .get_prediction_confidence_distribution(probabilities)
        )

        df = pd.DataFrame(
            list(prob_dist.items()),
            columns=["Disease", "Probability"]
        ).sort_values("Probability")

        fig, ax = plt.subplots()
        ax.barh(df["Disease"], df["Probability"])
        ax.set_xlabel("Probability")
        st.pyplot(fig)

    # ---------------------------------------------------
    # GRADCAM
    # ---------------------------------------------------

    st.divider()
    st.subheader("Grad-CAM Visualization")

    try:

        if st.session_state.gradcam is None:

            st.session_state.gradcam = GradCAMVisualizer(
                st.session_state.predictor.model
            )

        visualizer = st.session_state.gradcam

        class_idx = int(np.argmax(probabilities))

        heatmap = visualizer.generate_cam(image_array, class_idx)

        # normalize
        heatmap = np.nan_to_num(heatmap)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        heatmap = cv2.resize(
            heatmap,
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        )

        # FIX FOR OPENCV ERROR
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        heatmap_uint8 = np.array(heatmap_uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        heatmap_color = heatmap_color.astype(np.float32) / 255.0

        overlay = image_array * 0.6 + heatmap_color * 0.4

        overlay = np.clip(overlay, 0, 1)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_array, caption="Original")

        with col2:
            st.image(heatmap_color, caption="GradCAM Heatmap")

        with col3:
            st.image(overlay, caption="Overlay")

    except Exception as e:

        st.warning(f"GradCAM failed: {e}")

    st.session_state.history.append(
        {
            "timestamp": pd.Timestamp.now(),
            "disease": predicted_class,
            "confidence": confidence
        }
    )


# ---------------------------------------------------
# MODEL INFO
# ---------------------------------------------------

def show_model_info():

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Input Size", f"{Config.IMAGE_SIZE} x {Config.IMAGE_SIZE}")

    with col2:
        st.metric("Classes", Config.NUM_CLASSES)


# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------

def show_history():

    st.subheader("Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet")
        return

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

def show_about():

    st.subheader("About")

    st.write("""
Retinal Disease Classification System

This system detects diabetic retinopathy using deep learning.

Technologies used:

TensorFlow  
OpenCV  
Streamlit
""")


# ---------------------------------------------------

if __name__ == "__main__":
    main()