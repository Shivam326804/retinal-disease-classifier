"""
Model Builder Module
Constructs various CNN architectures for retinal disease classification
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
metrics = tf.keras.metrics
applications = tf.keras.applications


class ModelBuilder:
    """Builds CNN architectures for retinal disease classification"""

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 5):

        self.input_shape = input_shape
        self.num_classes = num_classes

        logger.info(
            f"ModelBuilder initialized: input_shape={input_shape}, num_classes={num_classes}"
        )

    # ---------------------------------------------------
    # BASELINE CNN
    # ---------------------------------------------------

    def build_baseline_cnn(self) -> tf.keras.Model:

        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy", metrics.AUC()]
        )

        logger.info("Baseline CNN model created")
        return model

    # ---------------------------------------------------
    # CUSTOM CNN
    # ---------------------------------------------------

    def build_custom_cnn(self) -> tf.keras.Model:

        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy", metrics.AUC()]
        )

        logger.info("Custom CNN model created")
        return model

    # ---------------------------------------------------
    # RESNET50
    # ---------------------------------------------------

    def build_resnet50(self) -> tf.keras.Model:

        base_model = applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.BatchNormalization()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(base_model.input, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy", metrics.AUC()]
        )

        logger.info("ResNet50 model created")
        return model

    # ---------------------------------------------------
    # EFFICIENTNET
    # ---------------------------------------------------

    def build_efficientnet(self) -> tf.keras.Model:

        base_model = applications.EfficientNetB4(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.BatchNormalization()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(base_model.input, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy", metrics.AUC()]
        )

        logger.info("EfficientNetB4 model created")
        return model

    # ---------------------------------------------------
    # INCEPTION V3
    # ---------------------------------------------------

    def build_inceptionv3(self) -> tf.keras.Model:

        base_model = applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(base_model.input, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy", metrics.AUC()]
        )

        logger.info("InceptionV3 model created")
        return model

    # ---------------------------------------------------
    # MODEL SUMMARY
    # ---------------------------------------------------

    def get_model_summary(self, model: tf.keras.Model) -> str:
        """Return model summary as string"""

        stream = io.StringIO()

        with redirect_stdout(stream):
            model.summary()

        return stream.getvalue()