"""
Prediction Module - FINAL CORRECT VERSION
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional
from pathlib import Path
import traceback

from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class Predictor:
    """Handles model prediction tasks"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        # ✅ FIX: dynamic model path
        self.model_path: str = str(model_path or Config.get_model_path())

        if isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = Config.DISEASE_CLASSES

        self.model: Optional[tf.keras.Model] = None

        self.load_model()

    # ---------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------
    def load_model(self) -> bool:
        try:
            logger.info(f"🔄 Loading model from: {self.model_path}")

            if not self.model_path or not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False
            )

            logger.info("✅ Model loaded successfully")

            # Warmup
            dummy = np.zeros(
                (1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
                dtype=np.float32
            )

            self.model.predict(dummy, verbose=0)

            return True

        except Exception as e:
            traceback.print_exc()
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.model = None
            return False

    # ---------------------------------------------------
    # PREPROCESS INPUT (CRITICAL FIX)
    # ---------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")

        # Resize if needed
        if image.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            image = tf.image.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE)).numpy()

        image = image.astype(np.float32)

        # ✅ IMPORTANT: EfficientNet preprocessing
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        return image

    # ---------------------------------------------------
    # SINGLE IMAGE PREDICTION
    # ---------------------------------------------------
    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            image = self.preprocess(image)

            predictions = self.model.predict(image, verbose=0)

            probabilities = predictions[0]

            predicted_class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class_idx])

            predicted_class = self.class_names.get(
                predicted_class_idx,
                f"Class {predicted_class_idx}"
            )

            logger.info(
                f"Prediction: {predicted_class} "
                f"(confidence: {confidence:.4f})"
            )

            return predicted_class, confidence, probabilities

        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            traceback.print_exc()

            return "Error", 0.0, np.zeros(len(self.class_names))

    # ---------------------------------------------------
    # BATCH PREDICTION
    # ---------------------------------------------------
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            images = np.array([self.preprocess(img)[0] for img in images])

            predictions = self.model.predict(images, verbose=0)

            logger.info(f"Batch prediction completed: {len(images)} images")

            return predictions

        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {str(e)}")
            traceback.print_exc()
            return np.array([])

    # ---------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # ---------------------------------------------------
    def get_prediction_confidence_distribution(
        self,
        probabilities: np.ndarray
    ) -> Dict[str, float]:

        distribution = {}

        try:
            for idx, prob in enumerate(probabilities):
                class_name = self.class_names.get(idx, f"Class {idx}")
                distribution[class_name] = float(prob)

            return distribution

        except Exception as e:
            logger.error(f"Distribution error: {str(e)}")
            return distribution