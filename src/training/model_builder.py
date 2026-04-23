import tensorflow as tf
from typing import Tuple
from contextlib import redirect_stdout
import io

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

layers = tf.keras.layers
models = tf.keras.models
applications = tf.keras.applications
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers


class ModelBuilder:

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (260, 260, 3),
        num_classes: int = 5
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = None

    # ---------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------
    def build_model(self) -> tf.keras.Model:

        print("🔄 Loading EfficientNetB3...")

        base_model = applications.EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        print("✅ EfficientNetB3 loaded")

        base_model.trainable = False
        self.base_model = base_model

        inputs = layers.Input(shape=self.input_shape)

        x = tf.cast(inputs, tf.float32)
        x = applications.efficientnet.preprocess_input(x)

        # Keep BN stable
        x = base_model(x, training=False)

        # HEAD
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)

        logger.info("🔥 Model built (EfficientNetB3 FINAL)")
        return model

    # ---------------------------------------------------
    # FINE TUNE
    # ---------------------------------------------------
    def fine_tune_model(self, model, unfreeze_layers=180):

        logger.info("🔓 Fine-tuning...")

        for layer in self.base_model.layers:
            layer.trainable = False

        for layer in self.base_model.layers[-unfreeze_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        logger.info(f"✅ Unfroze last {unfreeze_layers} layers")
        return model

    # ---------------------------------------------------
    # COMPILE
    # ---------------------------------------------------
    def compile_model(self, model, lr=1e-4, loss_fn=None):

        if loss_fn is None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        optimizer = optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")
            ]
        )

        logger.info(f"⚙️ Compiled (lr={lr})")
        return model

    # ---------------------------------------------------
    def get_model_summary(self, model):
        stream = io.StringIO()
        with redirect_stdout(stream):
            model.summary()
        return stream.getvalue()