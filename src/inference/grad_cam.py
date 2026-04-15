"""
Grad-CAM Visualization Module (FINAL PRODUCTION VERSION)
Handles nested models, augmentation pipelines, and graph mismatch
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import Optional, Any, cast
from matplotlib.figure import Figure

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

cv2 = cast(Any, cv2)


class GradCAMVisualizer:
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        self.model = model

        # 🔥 STEP 1: Find actual CNN model
        self.base_model = self._find_base_model(self.model)

        if self.base_model is None:
            raise ValueError("❌ Could not find CNN base model")

        # 🔥 STEP 2: Get last conv layer
        try:
            self.layer_name = layer_name or self._get_last_conv_layer()
        except Exception as e:
            raise ValueError(f"❌ {e}")

        # 🔥 STEP 3: Build grad model
        self.grad_model = None
        self._build_grad_model()

        if self.grad_model is None:
            raise RuntimeError("❌ Grad model initialization failed")

        logger.info(f"✅ GradCAM initialized using layer: {self.layer_name}")

    # ---------------------------------------------------
    # 🔥 FIND CNN MODEL (RECURSIVE)
    # ---------------------------------------------------
    def _find_base_model(self, model) -> Optional[tf.keras.Model]:
        # If current model has Conv2D → return it
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return model

        # Otherwise search deeper
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                found = self._find_base_model(layer)
                if found is not None:
                    return found

        return None

    # ---------------------------------------------------
    # 🔥 GET LAST CONV LAYER
    # ---------------------------------------------------
    def _get_last_conv_layer(self) -> str:
        for layer in reversed(self.base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("No Conv2D layer found in CNN model")

    # ---------------------------------------------------
    # 🔥 BUILD GRAD MODEL (FIXED GRAPH ISSUE)
    # ---------------------------------------------------
    def _build_grad_model(self):
        try:
            conv_layer = self.base_model.get_layer(self.layer_name)

            # 🔥 IMPORTANT: Use base_model graph
            self.grad_model = tf.keras.models.Model(
                inputs=self.base_model.input,
                outputs=[conv_layer.output, self.base_model.output],
            )

            logger.info("✅ Grad model built using base_model")

        except Exception as e:
            logger.error(f"❌ Grad model build failed: {str(e)}")
            self.grad_model = None

    # ---------------------------------------------------
    # 🔥 PASS IMAGE THROUGH PREPROCESSING
    # ---------------------------------------------------
    def _forward_to_base(self, image_tensor):
        """
        Pass input through preprocessing layers before base_model
        """
        x = image_tensor

        for layer in self.model.layers:
            if layer == self.base_model:
                break
            x = layer(x)

        return x

    # ---------------------------------------------------
    # 🔥 GENERATE GRAD-CAM
    # ---------------------------------------------------
    def generate_cam(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        try:
            if self.grad_model is None:
                raise RuntimeError("Grad model not initialized")

            # Ensure batch dimension
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

            # 🔥 Pass through preprocessing
            processed = self._forward_to_base(image_tensor)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(processed)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                raise RuntimeError("Gradients are None")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.nn.relu(heatmap)
            heatmap = heatmap.numpy()

            # Normalize
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            # 🔥 Enhance visibility
            heatmap = np.power(heatmap, 1.5)

            heatmap = cv2.resize(heatmap, (224, 224))

            return heatmap.astype(np.float32)

        except Exception as e:
            logger.error(f"❌ Grad-CAM generation failed: {str(e)}")
            return np.zeros((224, 224), dtype=np.float32)

    # ---------------------------------------------------
    # 🔥 OVERLAY
    # ---------------------------------------------------
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        try:
            if image.ndim == 4:
                image = image[0]

            h, w = image.shape[:2]

            heatmap = cv2.resize(heatmap, (w, h))
            heatmap = (heatmap * 255).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

            return overlay

        except Exception as e:
            logger.error(f"❌ Overlay failed: {str(e)}")
            return image

    # ---------------------------------------------------
    # OPTIONAL PLOT
    # ---------------------------------------------------
    @staticmethod
    def plot_gradcam(image, heatmap, overlay) -> Optional[Figure]:
        try:
            import matplotlib.pyplot as plt

            if image.ndim == 4:
                image = image[0]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"❌ Plot failed: {str(e)}")
            return None