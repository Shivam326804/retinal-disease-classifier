"""
Dataset Loader Module - FINAL HIGH-ACCURACY VERSION
"""

import numpy as np
import tensorflow as tf
from typing import Tuple
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class DatasetLoader:
    """Handles dataset loading, augmentation, and batching"""

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

        if augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    # ---------------------------------------------------
    # 🔥 STRONG AUGMENTATION (MEDICAL SAFE)
    # ---------------------------------------------------

    def _augment_image(self, image, label):

        # Horizontal flip (safe for retina)
        image = tf.image.random_flip_left_right(image)

        # Brightness / contrast
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Saturation (important for fundus images)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # Slight zoom (random crop + resize)
        scale = tf.random.uniform([], 0.85, 1.0)
        new_size = tf.cast(scale * self.image_size, tf.int32)

        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = tf.image.random_crop(image, size=[new_size, new_size, 3])
        image = tf.image.resize(image, (self.image_size, self.image_size))

        # Add slight noise (helps generalization)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
        image = image + noise

        image = tf.clip_by_value(image, 0.0, 1.0)

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

        # ✅ Ensure float32
        train_images = train_images.astype(np.float32)
        val_images = val_images.astype(np.float32)

        # ✅ Normalize
        if train_images.max() > 1.0:
            train_images /= 255.0
            val_images /= 255.0

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
    # 🔥 CLASS WEIGHTS (IMBALANCE FIX)
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
    # OPTIONAL: BALANCED SAMPLING (ADVANCED)
    # ---------------------------------------------------

    @staticmethod
    def balance_dataset(images, labels):
        """
        Oversample minority classes (optional)
        """

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

        logger.info("⚖️ Dataset balanced using oversampling")

        return balanced_images, balanced_labels