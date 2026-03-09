"""
Grad-CAM Visualization Module
Implements Grad-CAM for model interpretability
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import Optional
from matplotlib.figure import Figure

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GradCAMVisualizer:
    """Generates Grad-CAM visualizations for model predictions"""

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):

        self.model = model

        # Automatically detect last convolution layer
        self.layer_name = layer_name or self._get_last_conv_layer(model)

        self.grad_model: Optional[tf.keras.Model] = None

        self._build_grad_model()

        logger.info(f"GradCAM initialized for layer: {self.layer_name}")

    # ---------------------------------------------------
    # FIND LAST CONV LAYER
    # ---------------------------------------------------

    def _get_last_conv_layer(self, model: tf.keras.Model) -> str:
        """Find last convolutional layer automatically"""

        for layer in reversed(model.layers):

            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("No convolution layer found in model")

    # ---------------------------------------------------
    # BUILD GRAD MODEL
    # ---------------------------------------------------

    def _build_grad_model(self) -> None:
        """Create gradient model"""

        try:

            conv_layer = self.model.get_layer(self.layer_name)

            self.grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[conv_layer.output, self.model.output],
            )

        except Exception as e:

            logger.error(f"Error building gradient model: {str(e)}")
            self.grad_model = None

    # ---------------------------------------------------
    # GENERATE HEATMAP
    # ---------------------------------------------------

    def generate_cam(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap"""

        try:

            if self.grad_model is None:
                raise RuntimeError("Grad model not initialized")

            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

            with tf.GradientTape() as tape:

                conv_outputs, predictions = self.grad_model(image_tensor)

                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                raise RuntimeError("Gradients returned None")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

            heatmap = tf.maximum(heatmap, 0)

            heatmap = heatmap.numpy()

            # Normalize heatmap
            max_val = np.max(heatmap)

            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap

        except Exception as e:

            logger.error(f"Grad-CAM generation failed: {str(e)}")

            # fallback safe heatmap
            return np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

    # ---------------------------------------------------
    # OVERLAY HEATMAP
    # ---------------------------------------------------

    def visualize_with_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on image"""

        try:

            h, w = image.shape[:2]

            heatmap = cv2.resize(heatmap, (w, h))

            heatmap_uint8 = (heatmap * 255).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )

            heatmap_color = cv2.cvtColor(
                heatmap_color,
                cv2.COLOR_BGR2RGB,
            )

            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)

            overlay = cv2.addWeighted(
                image_uint8,
                1 - alpha,
                heatmap_color,
                alpha,
                0,
            )

            return overlay

        except Exception as e:

            logger.error(f"Grad-CAM overlay failed: {str(e)}")

            return image

    # ---------------------------------------------------
    # PLOT RESULTS
    # ---------------------------------------------------

    @staticmethod
    def plot_gradcam(
        image: np.ndarray,
        heatmap: np.ndarray,
        visualization: np.ndarray,
    ) -> Optional[Figure]:
        """Plot Grad-CAM results"""

        try:

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(visualization)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()

            return fig

        except Exception as e:

            logger.error(f"Grad-CAM plotting failed: {str(e)}")

            return None