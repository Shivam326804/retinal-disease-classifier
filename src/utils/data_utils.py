"""
Data Utility Functions
Handles file operations, image loading, and data validation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


def create_directories(paths: list) -> None:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created/verified: {path}")


def load_image(image_path: str, target_size: Tuple[int, int] = (300, 300)) -> Optional[np.ndarray]:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        target_size: Target image dimensions (height, width)
    
    Returns:
        Preprocessed image array or None if loading fails
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        logger.debug(f"Image loaded successfully: {image_path}")
        return img
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to disk
    
    Args:
        image: Image array
        output_path: Path to save image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8 if normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, image)
        logger.debug(f"Image saved successfully: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {str(e)}")
        return False


def get_image_files(directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """
    Get all image files from directory
    
    Args:
        directory: Directory path
        extensions: Image file extensions to search for
    
    Returns:
        List of image file paths
    """
    image_files = []
    try:
        for ext in extensions:
            image_files.extend(Path(directory).glob(f"**/*{ext}"))
            image_files.extend(Path(directory).glob(f"**/*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {directory}")
        return sorted([str(f) for f in image_files])
    
    except Exception as e:
        logger.error(f"Error reading directory {directory}: {str(e)}")
        return []


def validate_image(image_path: str) -> bool:
    """
    Validate if image file is readable
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        return img is not None
    except Exception as e:
        logger.error(f"Image validation failed for {image_path}: {str(e)}")
        return False


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size
    
    Args:
        file_path: Path to file
    
    Returns:
        Formatted file size string
    """
    try:
        size_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return "Unknown"
