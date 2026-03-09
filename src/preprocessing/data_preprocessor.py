"""
Data Preprocessor Module
Handles image preprocessing, normalization, and dataset preparation
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks"""

    def __init__(self, image_size: int = 224, normalize: bool = True):

        self.image_size = image_size
        self.normalize = normalize

        self.processed_data_dir = Path(Config.PROCESSED_DATA_DIR)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataPreprocessor initialized with image_size={image_size}")

    # ---------------------------------------------------
    # FUNDUS CROPPING
    # ---------------------------------------------------

    def crop_fundus(self, img: np.ndarray) -> np.ndarray:
        """Remove black borders around retina"""

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        coords = np.column_stack(np.where(thresh > 0))

        if coords.size == 0:
            return img

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        return img[y0:y1, x0:x1]

    # ---------------------------------------------------
    # BEN GRAHAM PREPROCESSING
    # ---------------------------------------------------

    def ben_graham_preprocess(self, img: np.ndarray, sigma: int = 10) -> np.ndarray:
        """Enhance retinal vessels"""

        return cv2.addWeighted(
            img,
            4,
            cv2.GaussianBlur(img, (0, 0), sigma),
            -4,
            128,
        )

    # ---------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------

    def load_image(
        self,
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:

        try:

            img = cv2.imread(str(image_path))

            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.crop_fundus(img)

            img = self.ben_graham_preprocess(img)

            if target_size is not None:
                h, w = target_size
                img = cv2.resize(img, (w, h))
            else:
                img = self._resize_with_padding(img, self.image_size)

            if self.normalize:
                img = img.astype(np.float32) / 255.0

            return img

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    # ---------------------------------------------------
    # RESIZE WITH PADDING
    # ---------------------------------------------------

    def _resize_with_padding(self, img: np.ndarray, target_size: int) -> np.ndarray:

        h, w = img.shape[:2]

        scale = min(target_size / h, target_size / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(img, (new_w, new_h))

        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        return padded

    # ---------------------------------------------------
    # DATASET PREPROCESSING
    # ---------------------------------------------------

    def preprocess_dataset(
        self,
        images_dir: Path,
        labels_df: Optional[pd.DataFrame] = None,
        save_processed: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        logger.info(f"Starting dataset preprocessing from {images_dir}")

        images = []
        labels = []

        failed_count = 0

        if labels_df is not None:

            for i, (_, row) in enumerate(labels_df.iterrows()):

                image_id = str(row["id_code"])
                label = int(row["diagnosis"])

                # SEARCH IMAGE INSIDE SUBFOLDERS
                matches = list(images_dir.rglob(f"{image_id}.png"))

                if not matches:
                    failed_count += 1
                    continue

                image_path = matches[0]

                img = self.load_image(image_path)

                if img is not None:
                    images.append(img)
                    labels.append(label)
                else:
                    failed_count += 1

                if (i + 1) % 200 == 0:
                    logger.info(f"Processed {i + 1} images")

        else:

            for image_path in images_dir.rglob("*"):

                if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                    img = self.load_image(image_path)

                    if img is not None:
                        images.append(img)

        images_array = np.array(images, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32) if labels else None

        logger.info(
            f"Preprocessing complete. Images: {len(images)}, Failed: {failed_count}"
        )

        if images_array.size == 0:
            logger.error("No images processed. Check dataset paths.")
            return images_array, labels_array

        logger.info(f"Images shape: {images_array.shape}")

        if save_processed:
            self.save_processed_data(images_array, labels_array)

        return images_array, labels_array

    # ---------------------------------------------------
    # DATASET SPLIT
    # ---------------------------------------------------

    def split_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        X_temp, X_test, y_temp, y_test = train_test_split(
            images,
            labels,
            test_size=test_ratio,
            random_state=random_state,
            stratify=labels,
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=y_temp,
        )

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        logger.info(
            f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
        )

        return splits

    # ---------------------------------------------------
    # SAVE DATA
    # ---------------------------------------------------

    def save_processed_data(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> None:

        try:

            np.save(self.processed_data_dir / "images.npy", images)

            logger.info("Saved processed images")

            if labels is not None:
                np.save(self.processed_data_dir / "labels.npy", labels)
                logger.info("Saved labels")

        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    def load_processed_data(self):

        try:

            images = np.load(self.processed_data_dir / "images.npy")
            labels = np.load(self.processed_data_dir / "labels.npy")

            logger.info(
                f"Loaded data. Images: {images.shape}, Labels: {labels.shape}"
            )

            return images, labels

        except FileNotFoundError:

            logger.warning("Processed dataset not found")

            return None, None

    # ---------------------------------------------------
    # STREAMLIT PREPROCESSING
    # ---------------------------------------------------

    def preprocess_image_array(self, image: np.ndarray) -> np.ndarray:

        img = image.copy()

        img = self.crop_fundus(img)

        img = self.ben_graham_preprocess(img)

        img = self._resize_with_padding(img, self.image_size)

        if self.normalize:
            img = img.astype(np.float32) / 255.0

        return img


# ---------------------------------------------------
# RUN PREPROCESSING
# ---------------------------------------------------

if __name__ == "__main__":

    logger.info("Starting preprocessing pipeline")

    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)

    images_dir = Config.IMAGES_DIR
    labels_file = Config.LABELS_FILE

    labels_df = pd.read_csv(labels_file)

    images, labels = preprocessor.preprocess_dataset(
        images_dir=images_dir,
        labels_df=labels_df,
        save_processed=True
    )

    logger.info("Preprocessing completed")

    if images is not None:
        logger.info(f"Processed images: {images.shape}")