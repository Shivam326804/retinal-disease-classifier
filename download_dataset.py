"""
Dataset Download Script
Automatically downloads and prepares retinal disease dataset
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.preprocessing.data_preprocessor import DataPreprocessor

logger = setup_logger(__name__)


def create_dummy_dataset(num_samples: int = 200) -> Path:
    """
    Create a dummy dataset for demonstration
    (In production, download from Kaggle or actual source)

    Args:
        num_samples: Number of images to create

    Returns:
        Path to dataset directory
    """

    logger.info(f"Creating dummy dataset with {num_samples} samples...")

    import cv2

    dataset_dir = Path(Config.RAW_DATA_DIR) / "APTOS_2019"
    train_dir = dataset_dir / "train"

    train_dir.mkdir(parents=True, exist_ok=True)

    labels_data = []

    for i in tqdm(range(num_samples), desc="Creating dummy images"):

        # Random image
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # Add circular pattern (simulate retina)
        cv2.circle(img, (256, 256), 100, (100, 150, 200), -1)
        cv2.circle(img, (256, 256), 80, (150, 180, 220), -1)

        filename = f"image_{i:05d}.jpg"
        filepath = train_dir / filename

        cv2.imwrite(str(filepath), img)

        label = np.random.randint(0, 5)

        labels_data.append(
            {
                "image_id": filename,
                "diagnosis": label,
            }
        )

    labels_df = pd.DataFrame(labels_data)

    labels_csv = dataset_dir / "train_labels.csv"
    labels_df.to_csv(labels_csv, index=False)

    logger.info(f"Dummy dataset created at {train_dir}")
    logger.info(f"Labels saved at {labels_csv}")

    return dataset_dir


def prepare_dataset(dataset_path: Path) -> None:
    """
    Prepare dataset for training

    Args:
        dataset_path: Path to raw dataset
    """

    logger.info(f"Preparing dataset from {dataset_path}...")

    labels_csv = dataset_path / "train_labels.csv"

    if labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        logger.info(f"Loaded {len(labels_df)} labels")
    else:
        logger.warning("No labels CSV found")
        labels_df = None

    images_dir = dataset_path / "train"

    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)

    logger.info("Preprocessing started...")

    # FIX: images_dir passed as Path instead of str
    images, labels = preprocessor.preprocess_dataset(
        images_dir=images_dir,
        labels_df=labels_df,
        save_processed=True,
    )

    logger.info("Preprocessing complete!")
    logger.info(f"Images shape: {images.shape}")

    if labels is not None:
        logger.info(f"Labels shape: {labels.shape}")

    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)

        logger.info("\nClass Distribution:")

        for cls_id, count in zip(unique, counts):
            logger.info(f"Class {cls_id}: {count} samples")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Download and prepare dataset")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to existing dataset (if None, creates dummy)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of dummy samples to create",
    )

    args = parser.parse_args()

    Config.create_all_directories()

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = create_dummy_dataset(args.num_samples)

    prepare_dataset(dataset_path)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()