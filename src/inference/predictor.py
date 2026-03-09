"""
Prediction Module
Handles model inference and prediction
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Predictor:
    """Handles model prediction tasks"""

    def __init__(self, model_path: str, class_names: Dict[int, str]):

        self.model_path: str = model_path
        self.class_names: Dict[int, str] = class_names
        self.model: Optional[tf.keras.Model] = None

        self.load_model()

    # ---------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------

    def load_model(self) -> bool:

        try:

            model = tf.keras.models.load_model(self.model_path)

            self.model = model

            logger.info(f"Model loaded from {self.model_path}")

            # Warmup inference
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)

            try:
                model.predict(dummy, verbose=0)
            except Exception:
                pass

            return True

        except Exception as e:

            logger.error(f"Error loading model: {str(e)}")

            self.model = None

            return False

    # ---------------------------------------------------
    # SINGLE IMAGE PREDICTION
    # ---------------------------------------------------

    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:

        try:

            if self.model is None:
                raise RuntimeError("Model not loaded")

            model = self.model  # FIX for Pylance

            if image is None:
                raise ValueError("Input image is None")

            image = image.astype(np.float32)

            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            if image.ndim != 4:
                raise ValueError("Invalid image shape")

            predictions = model.predict(image, verbose=0)

            probabilities: np.ndarray = predictions[0]

            predicted_class_idx: int = int(np.argmax(probabilities))

            confidence: float = float(probabilities[predicted_class_idx])

            predicted_class: str = self.class_names.get(
                predicted_class_idx,
                "Unknown"
            )

            logger.info(
                f"Prediction: {predicted_class} (confidence: {confidence:.4f})"
            )

            return predicted_class, confidence, probabilities

        except Exception as e:

            logger.error(f"Prediction failed: {str(e)}")

            return "Error", 0.0, np.zeros(len(self.class_names))

    # ---------------------------------------------------
    # BATCH PREDICTION
    # ---------------------------------------------------

    def predict_batch(self, images: np.ndarray) -> np.ndarray:

        try:

            if self.model is None:
                raise RuntimeError("Model not loaded")

            model = self.model  # FIX for Pylance

            if images is None or len(images) == 0:
                raise ValueError("Empty image batch")

            images = images.astype(np.float32)

            predictions = model.predict(images, verbose=0)

            logger.info(
                f"Batch prediction completed for {len(images)} images"
            )

            return predictions

        except Exception as e:

            logger.error(f"Batch prediction failed: {str(e)}")

            return np.array([])

    # ---------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # ---------------------------------------------------

    def get_prediction_confidence_distribution(
        self,
        probabilities: np.ndarray
    ) -> Dict[str, float]:

        distribution: Dict[str, float] = {}

        try:

            if probabilities is None or len(probabilities) == 0:
                return distribution

            for idx, prob in enumerate(probabilities):

                class_name: str = self.class_names.get(
                    idx,
                    f"Class {idx}"
                )

                distribution[class_name] = float(prob)

            return distribution

        except Exception as e:

            logger.error(
                f"Error computing probability distribution: {str(e)}"
            )

            return distribution