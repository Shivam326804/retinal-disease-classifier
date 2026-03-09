"""
Dataset Loader Module
Handles data loading for training, validation, and testing
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from pathlib import Path
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class DatasetLoader:
    """Handles dataset loading and batching"""
    
    def __init__(self, batch_size: int = 32, image_size: int = 300):
        """
        Initialize dataset loader
        
        Args:
            batch_size: Batch size for data loading
            image_size: Image size
        """
        self.batch_size = batch_size
        self.image_size = image_size
        logger.info(f"DatasetLoader initialized: batch_size={batch_size}, image_size={image_size}")
    
    def create_tf_dataset(self,
                         images: np.ndarray,
                         labels: np.ndarray,
                         augment: bool = False,
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from numpy arrays
        
        Args:
            images: Image array
            labels: Label array
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
        
        Returns:
            TensorFlow dataset
        """
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        if augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Created TF dataset with {len(images)} samples, batch_size={self.batch_size}")
        return dataset
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentation to image"""
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Clip to valid range
        image = tf.clip_by_value(image, 0, 1)
        
        return image, label
    
    @staticmethod
    def create_generators(train_images: np.ndarray,
                         train_labels: np.ndarray,
                         val_images: np.ndarray,
                         val_labels: np.ndarray,
                         batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create train and validation data generators
        
        Args:
            train_images: Training images
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels
            batch_size: Batch size
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Normalize if needed
        if train_images.max() > 1.0:
            train_images = train_images / 255.0
            val_images = val_images / 255.0
        
        loader = DatasetLoader(batch_size=batch_size)
        
        train_dataset = loader.create_tf_dataset(train_images, train_labels, augment=True, shuffle=True)
        val_dataset = loader.create_tf_dataset(val_images, val_labels, augment=False, shuffle=False)
        
        return train_dataset, val_dataset
    
    @staticmethod
    def get_class_weights(labels: np.ndarray) -> dict:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            labels: Label array
        
        Returns:
            Dictionary of class weights
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {}
        
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        
        logger.info(f"Class weights: {weights}")
        return weights
