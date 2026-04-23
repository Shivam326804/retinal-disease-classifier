import os
import random
import numpy as np
from pathlib import Path


class Config:
    """Central configuration for full pipeline (FINAL STABLE VERSION)"""

    # ---------------------------------------------------
    # BASE PATHS
    # ---------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    MODELS_DIR = (BASE_DIR / "models").resolve()
    LOGS_DIR = (BASE_DIR / "logs").resolve()
    REPORTS_DIR = (BASE_DIR / "reports").resolve()

    # ---------------------------------------------------
    # MODEL PATH
    # ---------------------------------------------------
    MODEL_PATH = (MODELS_DIR / "final_model.keras").resolve()

    @classmethod
    def get_model_path(cls):
        if cls.MODEL_PATH.exists():
            return cls.MODEL_PATH
        raise FileNotFoundError(f"❌ Model not found at: {cls.MODEL_PATH}")

    # ---------------------------------------------------
    # 📂 RAW DATA (APTOS)
    # ---------------------------------------------------
    DATASET_DIR = (BASE_DIR / "data" / "raw" / "APTOS_2019").resolve()

    IMAGES_DIR = (DATASET_DIR / "train_images").resolve()
    LABELS_FILE = (DATASET_DIR / "train.csv").resolve()

    # ---------------------------------------------------
    # 📂 PROCESSED DATA
    # ---------------------------------------------------
    DATA_DIR = (BASE_DIR / "data" / "processed").resolve()

    IMAGES_PATH = (DATA_DIR / "images.npy").resolve()
    LABELS_PATH = (DATA_DIR / "labels.npy").resolve()

    # Cache (for tf.data)
    CACHE_DIR = (DATA_DIR / "cache").resolve()

    @classmethod
    def get_data_paths(cls):
        if not cls.IMAGES_PATH.exists() or not cls.LABELS_PATH.exists():
            raise FileNotFoundError(
                f"❌ Dataset not found:\n{cls.IMAGES_PATH}\n{cls.LABELS_PATH}"
            )
        return cls.IMAGES_PATH, cls.LABELS_PATH

    # ---------------------------------------------------
    # 🔥 TRAINING SETTINGS (FINAL)
    # ---------------------------------------------------
    IMAGE_SIZE = 260   # ✅ MUST match EfficientNetB3
    BATCH_SIZE = 8
    VALIDATION_SPLIT = 0.2

    # ---------------------------------------------------
    # CLASSES
    # ---------------------------------------------------
    CLASS_NAMES = [
        "No DR",
        "Mild NPDR",
        "Moderate NPDR",
        "Severe NPDR",
        "Proliferative DR"
    ]

    NUM_CLASSES = len(CLASS_NAMES)

    # ---------------------------------------------------
    # 🔥 CLASS WEIGHTS (GLOBAL REFERENCE)
    # ---------------------------------------------------
    CLASS_WEIGHTS = {
        0: 0.6,
        1: 1.8,
        2: 0.9,
        3: 3.0,   # Severe boosted
        4: 2.5    # Proliferative boosted
    }

    # ---------------------------------------------------
    # LOGGER
    # ---------------------------------------------------
    LOG_LEVEL = "INFO"

    # ---------------------------------------------------
    # APP
    # ---------------------------------------------------
    APP_NAME = "Diabetic Retinopathy Screening AI"
    APP_VERSION = "1.0"

    # ---------------------------------------------------
    # 🔥 REPRODUCIBILITY
    # ---------------------------------------------------
    RANDOM_SEED = 42

    @staticmethod
    def set_seed(seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except:
            pass

    # ---------------------------------------------------
    # DIRECTORY MANAGEMENT
    # ---------------------------------------------------
    @classmethod
    def create_directories(cls):
        for path in [
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
            cls.DATA_DIR,
            cls.CACHE_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------
    @classmethod
    def validate_setup(cls):

        print("\n🔍 Validating configuration...\n")

        if not cls.IMAGES_DIR.exists():
            raise FileNotFoundError(f"❌ Images dir missing: {cls.IMAGES_DIR}")

        if not cls.LABELS_FILE.exists():
            raise FileNotFoundError(f"❌ Labels CSV missing: {cls.LABELS_FILE}")

        print("✅ Config validated")
        print(f"📂 Images dir: {cls.IMAGES_DIR}")
        print(f"📄 Labels file: {cls.LABELS_FILE}")
        print(f"🖼 Image size: {cls.IMAGE_SIZE}")
        print(f"📊 Classes: {cls.CLASS_NAMES}")