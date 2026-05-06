import os
import random
import numpy as np
from pathlib import Path


class Config:
    """Central configuration (FINAL PRODUCTION VERSION)"""

    # ---------------------------------------------------
    # BASE PATHS
    # ---------------------------------------------------
    BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent.parent))

    MODELS_DIR = (BASE_DIR / "models").resolve()
    LOGS_DIR = (BASE_DIR / "logs").resolve()
    REPORTS_DIR = (BASE_DIR / "reports").resolve()

    # ---------------------------------------------------
    # MODEL PATH (ENV SAFE)
    # ---------------------------------------------------
    DEFAULT_MODEL_NAME = "final_model.keras"

    @classmethod
    def get_model_path(cls):
        """
        Priority:
        1. ENV variable (Render safe)
        2. Default local path
        """

        env_path = os.getenv("MODEL_PATH")

        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
            else:
                raise FileNotFoundError(f"❌ ENV MODEL_PATH not found: {env_path}")

        default_path = (cls.MODELS_DIR / cls.DEFAULT_MODEL_NAME).resolve()

        if default_path.exists():
            return default_path

        raise FileNotFoundError(f"❌ Model not found at: {default_path}")

    # ---------------------------------------------------
    # 📂 RAW DATA (TRAINING ONLY)
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

    CACHE_DIR = (DATA_DIR / "cache").resolve()

    @classmethod
    def get_data_paths(cls):
        if not cls.IMAGES_PATH.exists() or not cls.LABELS_PATH.exists():
            raise FileNotFoundError(
                f"❌ Dataset not found:\n{cls.IMAGES_PATH}\n{cls.LABELS_PATH}"
            )
        return cls.IMAGES_PATH, cls.LABELS_PATH

    # ---------------------------------------------------
    # 🔥 TRAINING SETTINGS
    # ---------------------------------------------------
    IMAGE_SIZE = 260
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
    # CLASS WEIGHTS
    # ---------------------------------------------------
    CLASS_WEIGHTS = {
        0: 0.6,
        1: 1.8,
        2: 0.9,
        3: 3.0,
        4: 2.5
    }

    # ---------------------------------------------------
    # LOGGER
    # ---------------------------------------------------
    LOG_LEVEL = "INFO"

    # ---------------------------------------------------
    # APP INFO
    # ---------------------------------------------------
    APP_NAME = "Diabetic Retinopathy Screening AI"
    APP_VERSION = "1.1"

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
    # VALIDATION (TRAINING ONLY)
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