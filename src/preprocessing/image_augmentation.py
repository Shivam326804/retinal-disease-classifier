"""
Image Augmentation Module
Handles data augmentation using Albumentations
Optimized for retinal fundus image classification
"""

import numpy as np
import albumentations as A
from typing import Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageAugmenter:
    """Provides augmentation pipelines"""

    def __init__(self):
        logger.info("ImageAugmenter initialized")

    # ---------------------------------------------------
    # TRAIN AUGMENTATIONS
    # ---------------------------------------------------

    @staticmethod
    def get_train_augmentations(image_size: int = 224) -> A.Compose:
        """
        Augmentations for training
        """

        transforms = [

            A.Resize(height=image_size, width=image_size),

            A.HorizontalFlip(p=0.5),

            A.Rotate(limit=25, p=0.5),

            # Replacement for deprecated ShiftScaleRotate
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-20, 20),
                p=0.5
            ),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.CLAHE(clip_limit=2, p=1.0),
                A.RandomGamma(p=1.0),
            ], p=0.4),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),

            # Cutout replacement
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.2
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]

        return A.Compose(transforms)

    # ---------------------------------------------------
    # VALIDATION AUGMENTATIONS
    # ---------------------------------------------------

    @staticmethod
    def get_val_augmentations(image_size: int = 224) -> A.Compose:
        """
        Validation augmentation
        """

        transforms = [

            A.Resize(height=image_size, width=image_size),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]

        return A.Compose(transforms)

    # ---------------------------------------------------
    # TEST AUGMENTATIONS
    # ---------------------------------------------------

    @staticmethod
    def get_test_augmentations(image_size: int = 224) -> A.Compose:
        """
        Test augmentation
        """

        transforms = [

            A.Resize(height=image_size, width=image_size),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]

        return A.Compose(transforms)

    # ---------------------------------------------------
    # LIGHTWEIGHT AUGMENTATIONS
    # ---------------------------------------------------

    @staticmethod
    def lightweight_augmentations(image_size: int = 224) -> A.Compose:
        """
        Faster augmentation pipeline
        """

        transforms = [

            A.Resize(height=image_size, width=image_size),

            A.HorizontalFlip(p=0.5),

            A.Rotate(limit=15, p=0.3),

            A.RandomBrightnessContrast(p=0.3),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]

        return A.Compose(transforms)

    # ---------------------------------------------------
    # AUGMENT SINGLE IMAGE
    # ---------------------------------------------------

    @staticmethod
    def augment_image(
        image: np.ndarray,
        transform: A.Compose
    ) -> np.ndarray:
        """
        Apply augmentation to single image
        """

        augmented = transform(image=image)

        return augmented["image"]

    # ---------------------------------------------------
    # AUGMENT BATCH
    # ---------------------------------------------------

    @staticmethod
    def augment_batch(
        images: np.ndarray,
        transform: A.Compose
    ) -> np.ndarray:
        """
        Apply augmentation to batch of images
        """

        augmented_images = []

        for img in images:

            augmented = transform(image=img)

            augmented_images.append(augmented["image"])

        return np.array(augmented_images, dtype=np.float32)

    # ---------------------------------------------------
    # TEST AUGMENTATION
    # ---------------------------------------------------

    def test_augmentation(self, image: np.ndarray) -> np.ndarray:

        transform = self.get_train_augmentations()

        augmented = transform(image=image)

        logger.info("Augmentation test successful")

        return augmented["image"]