"""
Data Utility Functions
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any, cast

from .logger import setup_logger
from .config import Config

logger = setup_logger(__name__)

cv2 = cast(Any, cv2)


# ---------------------------------------------------
# CREATE DIRECTORIES
# ---------------------------------------------------

def create_directories(paths: list) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# LOAD IMAGE (FIXED)
# ---------------------------------------------------

def load_image(
    image_path: str,
    target_size: Tuple[int, int] = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
) -> Optional[np.ndarray]:

    try:
        img = cv2.imread(image_path)

        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (target_size[1], target_size[0]))

        # ❌ REMOVED: /255 normalization
        # Let model handle preprocessing

        img = img.astype(np.float32)

        return img

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


# ---------------------------------------------------
# SAVE IMAGE
# ---------------------------------------------------

def save_image(image: np.ndarray, output_path: str) -> bool:

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # If normalized image, convert back
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, image)

        return True

    except Exception as e:
        logger.error(f"Error saving image {output_path}: {str(e)}")
        return False


# ---------------------------------------------------
# GET IMAGE FILES
# ---------------------------------------------------

def get_image_files(
    directory: str,
    extensions: tuple = ('.jpg', '.jpeg', '.png')
) -> list:

    image_files = []

    try:
        for ext in extensions:
            image_files.extend(Path(directory).glob(f"**/*{ext}"))
            image_files.extend(Path(directory).glob(f"**/*{ext.upper()}"))

        return sorted([str(f) for f in image_files])

    except Exception as e:
        logger.error(f"Error reading directory {directory}: {str(e)}")
        return []


# ---------------------------------------------------
# VALIDATE IMAGE
# ---------------------------------------------------

def validate_image(image_path: str) -> bool:
    try:
        img = cv2.imread(image_path)
        return img is not None
    except Exception:
        return False


# ---------------------------------------------------
# FILE SIZE
# ---------------------------------------------------

def get_file_size(file_path: str) -> str:
    try:
        size_bytes = os.path.getsize(file_path)

        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

        return f"{size_bytes:.2f} TB"

    except Exception:
        return "Unknown"