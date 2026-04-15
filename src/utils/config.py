"""
Configuration Module - FINAL STABLE VERSION
"""

import os
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the entire project"""

    # ---------------------------------------------------
    # BASE PATHS
    # ---------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    DATA_DIR = (BASE_DIR / "data").resolve()
    RAW_DATA_DIR = (DATA_DIR / "raw").resolve()
    PROCESSED_DATA_DIR = (DATA_DIR / "processed").resolve()

    MODELS_DIR = (BASE_DIR / "models").resolve()
    CHECKPOINTS_DIR = (MODELS_DIR / "checkpoints").resolve()

    LOGS_DIR = (BASE_DIR / "logs").resolve()
    REPORTS_DIR = (BASE_DIR / "reports").resolve()

    # ---------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

    # ---------------------------------------------------
    # REPRODUCIBILITY
    # ---------------------------------------------------
    @staticmethod
    def set_seed(seed: int = 42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except Exception:
            pass

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------
    DATASET_NAME = os.getenv("DATASET_NAME", "APTOS_2019")

    DATASET_PATH = Path(
        os.getenv("DATASET_PATH", str(RAW_DATA_DIR / DATASET_NAME))
    ).resolve()

    IMAGES_DIR = (DATASET_PATH / "train_images").resolve()
    LABELS_FILE = (DATASET_PATH / "train.csv").resolve()

    TEST_IMAGES_DIR = (DATASET_PATH / "test_images").resolve()
    TEST_CSV = (DATASET_PATH / "test.csv").resolve()

    # ---------------------------------------------------
    # PROCESSED FILES
    # ---------------------------------------------------
    PROCESSED_IMAGES = (PROCESSED_DATA_DIR / "images.npy").resolve()
    PROCESSED_LABELS = (PROCESSED_DATA_DIR / "labels.npy").resolve()

    # ---------------------------------------------------
    # MODEL PATHS (MATCH TRAINING)
    # ---------------------------------------------------
    BEST_MODEL_PATH = (CHECKPOINTS_DIR / "best_model.keras").resolve()
    FINAL_MODEL_PATH = (CHECKPOINTS_DIR / "final_model.keras").resolve()

    @classmethod
    def get_model_path(cls):
        """Get best available model safely"""
        if cls.BEST_MODEL_PATH.exists():
            return cls.BEST_MODEL_PATH

        if cls.FINAL_MODEL_PATH.exists():
            return cls.FINAL_MODEL_PATH

        # fallback: latest model
        model_files = list(cls.CHECKPOINTS_DIR.glob("*.keras"))
        if model_files:
            return sorted(model_files)[-1]

        return None

    # ---------------------------------------------------
    # TRAINING SETTINGS
    # ---------------------------------------------------
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    EPOCHS = int(os.getenv("EPOCHS", 40))

    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
    FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", 1e-5))

    EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 6))
    VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))

    # ---------------------------------------------------
    # CLASSES
    # ---------------------------------------------------
    DISEASE_CLASSES = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    CLASS_NAMES = list(DISEASE_CLASSES.values())
    NUM_CLASSES = len(DISEASE_CLASSES)

    # ---------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------
    @classmethod
    def create_all_directories(cls):
        """Create all required directories"""
        for directory in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_dataset(cls):
        """Validate dataset structure"""
        if not cls.IMAGES_DIR.exists():
            raise FileNotFoundError(f"❌ Images dir not found: {cls.IMAGES_DIR}")

        if not cls.LABELS_FILE.exists():
            raise FileNotFoundError(f"❌ CSV not found: {cls.LABELS_FILE}")

        image_count = len(list(cls.IMAGES_DIR.glob("*.png")))
        if image_count == 0:
            raise ValueError("❌ No images found in dataset directory")

        print(f"✅ Dataset verified ({image_count} images)")

    @classmethod
    def validate_processed_data(cls):
        """Validate processed dataset"""
        if not cls.PROCESSED_IMAGES.exists():
            raise FileNotFoundError("❌ Processed images not found")

        if not cls.PROCESSED_LABELS.exists():
            raise FileNotFoundError("❌ Processed labels not found")

        images = np.load(cls.PROCESSED_IMAGES)
        labels = np.load(cls.PROCESSED_LABELS)

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("✅ Processed data verified")

    @classmethod
    def validate_model(cls):
        """Validate trained model"""
        model_path = cls.get_model_path()

        if model_path is None:
            raise FileNotFoundError("❌ No trained model found")

        print(f"✅ Using model: {model_path}")
        return model_path


# ---------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------
Config.create_all_directories()
Config.set_seed(Config.RANDOM_SEED)