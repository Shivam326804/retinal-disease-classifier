import numpy as np
import tensorflow as tf
import cv2


class GradCAMVisualizer:
    """
    Stable Grad-CAM for models with backbone + custom head
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

        # ---------------------------------------------------
        # Extract EfficientNet backbone
        # ---------------------------------------------------
        self.base_model = self._get_base_model()

        # Last conv layer (EfficientNet standard)
        self.target_layer = self.base_model.get_layer("top_conv")

        print(f"✅ Backbone: {self.base_model.name}")
        print(f"✅ Target layer: {self.target_layer.name}")

        # ---------------------------------------------------
        # Backbone model (input → conv + features)
        # ---------------------------------------------------
        self.backbone_model = tf.keras.models.Model(
            inputs=self.base_model.input,
            outputs=[
                self.target_layer.output,
                self.base_model.output
            ]
        )

        # ---------------------------------------------------
        # Build classifier head (VERY IMPORTANT FIX)
        # ---------------------------------------------------
        x = self.base_model.output
        head_layers = []

        start = False
        for layer in self.model.layers:
            if layer.name == self.base_model.name:
                start = True
                continue
            if start:
                head_layers.append(layer)

        inp = tf.keras.Input(shape=x.shape[1:])
        y = inp
        for layer in head_layers:
            y = layer(y)

        self.classifier_model = tf.keras.models.Model(inp, y)

    # ---------------------------------------------------
    def _get_base_model(self):
        for layer in self.model.layers:
            if "efficientnet" in layer.name.lower():
                return layer
        raise ValueError("EfficientNet backbone not found")

    # ---------------------------------------------------
    def generate_cam(self, image, class_idx=None):

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        image = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:

            conv_outputs, features = self.backbone_model(image)
            tape.watch(conv_outputs)

            predictions = self.classifier_model(features)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None — check model graph")

        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(1, 2))

        conv_outputs = conv_outputs[0]
        weights = weights[0]

        cam = tf.reduce_sum(conv_outputs * weights, axis=-1)

        cam = tf.nn.relu(cam)

        # Normalize safely
        cam = cam.numpy()
        cam_min, cam_max = cam.min(), cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    # ---------------------------------------------------
    def overlay_heatmap(self, image, heatmap, alpha=0.5):

        if image.ndim == 4:
            image = image[0]

        h, w = image.shape[:2]

        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

        return overlay