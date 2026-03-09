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
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            class_names: Dictionary mapping class indices to names
        """

        self.model_path: str = model_path
        self.class_names: Dict[int, str] = class_names
        self.model: Optional[tf.keras.Model] = None

        self.load_model()

    def load_model(self) -> bool:
        """Load trained model"""

        try:

            self.model = tf.keras.models.load_model(self.model_path)

            logger.info(f"Model loaded from {self.model_path}")

            return True

        except Exception as e:

            logger.error(f"Error loading model: {str(e)}")

            self.model = None

            return False

    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict disease class for input image

        Args:
            image: Input image (H, W, C) normalized to [0, 1]

        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """

        try:

            if self.model is None:
                raise RuntimeError("Model not loaded")

            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)

            image = image.astype(np.float32)

            predictions = self.model.predict(image, verbose=0)

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

            logger.error(f"Error during prediction: {str(e)}")

            return "Error", 0.0, np.array([])

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Predict for batch of images

        Args:
            images: Batch of images (N, H, W, C)

        Returns:
            Batch of predictions
        """

        try:

            if self.model is None:
                raise RuntimeError("Model not loaded")

            images = images.astype(np.float32)

            predictions = self.model.predict(images, verbose=0)

            logger.info(
                f"Batch prediction completed for {len(images)} images"
            )

            return predictions

        except Exception as e:

            logger.error(f"Error during batch prediction: {str(e)}")

            return np.array([])

    def get_prediction_confidence_distribution(
        self,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        Get confidence distribution for all classes

        Args:
            probabilities: Prediction probabilities

        Returns:
            Dictionary of class -> confidence
        """

        distribution: Dict[str, float] = {}

        for idx, prob in enumerate(probabilities):

            idx_int: int = int(idx)

            class_name: str = self.class_names.get(
                idx_int,
                f"Class {idx_int}"
            )

            distribution[class_name] = float(prob)

        return distribution