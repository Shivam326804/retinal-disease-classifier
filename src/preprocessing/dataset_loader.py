import numpy as np
import tensorflow as tf
from typing import Tuple

from tensorflow.keras.applications.efficientnet import preprocess_input

from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class DatasetLoader:
    """Handles dataset loading, augmentation, and batching (final optimized version)"""

    def __init__(self, batch_size: int = 32, image_size: int = 224):
        self.batch_size = batch_size
        self.image_size = image_size

        logger.info(
            f"DatasetLoader initialized: batch_size={batch_size}, image_size={image_size}"
        )

    # ---------------------------------------------------
    # DATASET CREATION
    # ---------------------------------------------------

    def create_tf_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        shuffle: bool = True
    ) -> tf.data.Dataset:

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(len(images), 1000),
                reshuffle_each_iteration=True
            )

        dataset = dataset.map(
            self._preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if augment:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    # ---------------------------------------------------
    # 🔥 PREPROCESS (CORRECT FOR EFFICIENTNET)
    # ---------------------------------------------------

    def _preprocess(self, image, label):

        image = tf.cast(image, tf.float32)

        # EfficientNet preprocessing (VERY IMPORTANT)
        image = preprocess_input(image)

        return image, label

    # ---------------------------------------------------
    # 🔥 MEDICAL SAFE AUGMENTATION
    # ---------------------------------------------------

    def _augment(self, image, label):

        # Flip (safe)
        image = tf.image.random_flip_left_right(image)

        # Mild brightness/contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

        # Slight zoom
        scale = tf.random.uniform([], 0.9, 1.0)
        new_size = tf.cast(scale * self.image_size, tf.int32)

        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = tf.image.random_crop(image, size=[new_size, new_size, 3])
        image = tf.image.resize(image, (self.image_size, self.image_size))

        return image, label

    # ---------------------------------------------------
    # GENERATORS
    # ---------------------------------------------------

    @staticmethod
    def create_generators(
        train_images,
        train_labels,
        val_images,
        val_labels,
        batch_size=32
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        loader = DatasetLoader(
            batch_size=batch_size,
            image_size=Config.IMAGE_SIZE
        )

        train_dataset = loader.create_tf_dataset(
            train_images,
            train_labels,
            augment=True,
            shuffle=True
        )

        val_dataset = loader.create_tf_dataset(
            val_images,
            val_labels,
            augment=False,
            shuffle=False
        )

        return train_dataset, val_dataset

    # ---------------------------------------------------
    # CLASS WEIGHTS
    # ---------------------------------------------------

    @staticmethod
    def get_class_weights(labels: np.ndarray) -> dict:

        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        weights = {
            int(cls): float(total / (len(unique) * count))
            for cls, count in zip(unique, counts)
        }

        logger.info(f"🔥 Class weights: {weights}")

        return weights

    # ---------------------------------------------------
    # OPTIONAL BALANCING
    # ---------------------------------------------------

    @staticmethod
    def balance_dataset(images, labels):

        unique_classes = np.unique(labels)
        max_count = max([np.sum(labels == c) for c in unique_classes])

        balanced_images = []
        balanced_labels = []

        for c in unique_classes:
            class_images = images[labels == c]
            class_labels = labels[labels == c]

            repeat_factor = max_count // len(class_images) + 1

            class_images = np.tile(class_images, (repeat_factor, 1, 1, 1))[:max_count]
            class_labels = np.tile(class_labels, repeat_factor)[:max_count]

            balanced_images.append(class_images)
            balanced_labels.append(class_labels)

        balanced_images = np.concatenate(balanced_images, axis=0)
        balanced_labels = np.concatenate(balanced_labels, axis=0)

        logger.info("⚖️ Dataset balanced")

        return balanced_images, balanced_labels