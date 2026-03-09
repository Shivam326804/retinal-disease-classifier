"""
Configuration Management Module
Handles all project configuration settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------

load_dotenv()


class Config:
    """Configuration class for the entire project"""

    # ---------------------------------------------------
    # BASE PATHS
    # ---------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

    LOGS_DIR = BASE_DIR / "logs"
    REPORTS_DIR = BASE_DIR / "reports"

    # ---------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------

    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------

    DATASET_NAME = os.getenv("DATASET_NAME", "APTOS_2019")

    DATASET_PATH = Path(
        os.getenv("DATASET_PATH", str(RAW_DATA_DIR / DATASET_NAME))
    )

    IMAGES_DIR = DATASET_PATH / "colored_images"
    LABELS_FILE = DATASET_PATH / "train.csv"

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "retinal_classifier_efficientnet.keras"
    )

    MODEL_FULL_PATH = MODELS_DIR / MODEL_NAME

    # ---------------------------------------------------
    # TRAINING SETTINGS
    # ---------------------------------------------------

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    EPOCHS = int(os.getenv("EPOCHS", 100))

    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))

    EARLY_STOPPING_PATIENCE = int(
        os.getenv("EARLY_STOPPING_PATIENCE", 10)
    )

    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))

    VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
    TEST_SPLIT = float(os.getenv("TEST_SPLIT", 0.1))

    # ---------------------------------------------------
    # DISEASE CLASSES
    # ---------------------------------------------------

    DISEASE_CLASSES = {
        0: "No DR (Diabetic Retinopathy)",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    NUM_CLASSES = len(DISEASE_CLASSES)

    # ---------------------------------------------------
    # STREAMLIT
    # ---------------------------------------------------

    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

    # ---------------------------------------------------
    # API (OPTIONAL FUTURE EXTENSION)
    # ---------------------------------------------------

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 4))

    # ---------------------------------------------------
    # DATABASE (OPTIONAL)
    # ---------------------------------------------------

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./prediction_logs.db"
    )

    # ---------------------------------------------------
    # MODEL ARCHITECTURES
    # ---------------------------------------------------

    MODELS_TO_TRAIN = [
        "baseline_cnn",
        "custom_cnn",
        "resnet50",
        "efficientnet",
        "inceptionv3",
    ]

    # ---------------------------------------------------
    # DIRECTORY CREATION
    # ---------------------------------------------------

    @classmethod
    def create_all_directories(cls):
        """Create required project directories"""

        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # CONFIG EXPORT
    # ---------------------------------------------------

    @classmethod
    def get_config_dict(cls):
        """Return configuration dictionary"""

        return {
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
            "learning_rate": cls.LEARNING_RATE,
            "image_size": cls.IMAGE_SIZE,
            "num_classes": cls.NUM_CLASSES,
            "validation_split": cls.VALIDATION_SPLIT,
            "test_split": cls.TEST_SPLIT,
            "model_name": cls.MODEL_NAME,
        }


# ---------------------------------------------------
# INITIALIZE DIRECTORIES
# ---------------------------------------------------

Config.create_all_directories()


if __name__ == "__main__":

    print("Configuration loaded successfully")

    print(f"Base directory: {Config.BASE_DIR}")
    print(f"Dataset path: {Config.DATASET_PATH}")
    print(f"Model path: {Config.MODEL_FULL_PATH}")