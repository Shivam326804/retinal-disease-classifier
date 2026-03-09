"""
Training Module
Handles model training, validation, and checkpointing
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class Trainer:
    """Handles model training pipeline"""

    def __init__(self, model_name: str = "efficientnet", checkpoint_dir: Optional[str] = None):

        self.model_name = model_name

        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path(Config.CHECKPOINTS_DIR)
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_history: Dict[str, List[float]] = {}

        logger.info(f"Trainer initialized for model: {model_name}")

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------

    def train(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int,
        class_weights: Optional[Dict] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> Dict:

        if callbacks is None:
            callbacks = self._get_default_callbacks()

        logger.info("Starting training")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        self.training_history = history.history

        logger.info("Training completed")

        return history.history

    # ---------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------

    def _get_default_callbacks(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_{timestamp}.keras"

        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1,
            ),

            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path(Config.LOGS_DIR) / self.model_name),
                histogram_freq=1,
            ),
        ]

        return callbacks

    # ---------------------------------------------------
    # SAVE HISTORY
    # ---------------------------------------------------

    def save_training_history(self):

        history_path = Path(Config.LOGS_DIR) / f"{self.model_name}_history.json"

        history_path.parent.mkdir(parents=True, exist_ok=True)

        history_dict = {
            key: [float(v) for v in values]
            for key, values in self.training_history.items()
        }

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)

        logger.info(f"Training history saved to {history_path}")

    # ---------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------

    def save_model(self, model: tf.keras.Model, model_path: str) -> None:
        """Save trained model"""
        try:
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")


# ======================================================
# MAIN TRAINING PIPELINE
# ======================================================

if __name__ == "__main__":

    from src.training.model_builder import ModelBuilder

    print("\nStarting retinal disease model training\n")

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    images_path = Path(Config.PROCESSED_DATA_DIR) / "images.npy"
    labels_path = Path(Config.PROCESSED_DATA_DIR) / "labels.npy"

    if not images_path.exists() or not labels_path.exists():
        raise RuntimeError("Processed dataset not found. Run preprocessing first.")

    images = np.load(images_path)
    labels = np.load(labels_path)

    print("Dataset loaded:", images.shape)

    # ---------------------------------------------------
    # ONE HOT ENCODE
    # ---------------------------------------------------

    labels_categorical = tf.keras.utils.to_categorical(
        labels,
        num_classes=Config.NUM_CLASSES
    )

    # ---------------------------------------------------
    # DATA SPLIT
    # ---------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels_categorical,
        test_size=0.3,
        stratify=labels,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # ---------------------------------------------------
    # DATA AUGMENTATION
    # ---------------------------------------------------

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # ---------------------------------------------------
    # TF.DATA PIPELINE
    # ---------------------------------------------------

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000)

    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(Config.BATCH_SIZE)

    # ---------------------------------------------------
    # CLASS WEIGHTS
    # ---------------------------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = dict(enumerate(class_weights))

    print("Class weights:", class_weights)

    # ---------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------

    builder = ModelBuilder(
        input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
        num_classes=Config.NUM_CLASSES,
    )

    model = builder.build_efficientnet()

    print("\nModel Summary\n")
    model.summary()

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------

    trainer = Trainer(model_name="efficientnet")

    history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=Config.EPOCHS,
        class_weights=class_weights,
    )

    trainer.save_training_history()

    # ---------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------

    print("\nEvaluating model...\n")

    test_results = model.evaluate(test_dataset)

    print("\nTest Results:", test_results)

    # ---------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------

    model_path = Path(Config.MODELS_DIR) / "retinal_classifier_efficientnet.keras"

    model.save(model_path)

    print("\nModel saved to:", model_path)