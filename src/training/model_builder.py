"""
Model Builder Module - FINAL STABLE + HIGH-ACCURACY VERSION
"""

import tensorflow as tf
from typing import Tuple
from contextlib import redirect_stdout
import io

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
applications = tf.keras.applications


# ---------------------------------------------------
# 🔥 FIXED FOCAL LOSS (NO SHAPE BUG)
# ---------------------------------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):

        # ✅ ensure shape = (batch,)
        y_true = tf.squeeze(y_true)

        # ✅ convert to int
        y_true = tf.cast(y_true, tf.int32)

        # ✅ one-hot encode correctly
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        # Clip predictions (prevents NaN)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        ce = -y_true * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)

        prob = tf.reduce_sum(y_true * y_pred, axis=-1)

        loss_val = alpha * tf.pow(1 - prob, gamma) * ce

        return tf.reduce_mean(loss_val)

    return loss


# ---------------------------------------------------
# MODEL BUILDER
# ---------------------------------------------------

class ModelBuilder:
    """Builds CNN architectures for retinal disease classification"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 5
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # ---------------------------------------------------
    # 🔥 STAGE 1 MODEL (FROZEN BACKBONE)
    # ---------------------------------------------------

    def build_efficientnet(self) -> tf.keras.Model:
        """Stage 1: Train classification head only"""

        base_model = applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        # ✅ freeze backbone
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        for layer in base_model.layers[-50:]:
            layer.trainable = True

        x = base_model.output

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation="softmax"
        )(x)

        model = models.Model(
            inputs=base_model.input,
            outputs=outputs
        )

        # ✅ compile with FIXED focal loss
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss(),
            metrics=["accuracy"]
        )

        logger.info("🔥 Stage-1 EfficientNet created (frozen backbone)")

        return model

    # ---------------------------------------------------
    # 🔥 STAGE 2: FINE-TUNING
    # ---------------------------------------------------

    def fine_tune_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Stage 2: Unfreeze top layers"""

        logger.info("🔓 Starting fine-tuning...")

        # Unfreeze last 30 layers
        for layer in model.layers[-30:]:
            layer.trainable = True

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss=focal_loss(),
            metrics=["accuracy"]
        )

        return model

    # ---------------------------------------------------
    # MODEL SUMMARY
    # ---------------------------------------------------

    def get_model_summary(self, model: tf.keras.Model) -> str:
        stream = io.StringIO()

        with redirect_stdout(stream):
            model.summary()

        return stream.getvalue()