"""
Configuration Management Module
Handles all project configuration settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the entire project"""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    LOGS_DIR = BASE_DIR / "logs"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Environment
    DEBUG = os.getenv("DEBUG", "True") == "True"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Dataset
    DATASET_PATH = os.getenv("DATASET_PATH", str(RAW_DATA_DIR))
    DATASET_NAME = os.getenv("DATASET_NAME", "APTOS_2019")
    
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "retinal_classifier_efficientnet.keras")
    MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR))
    
    # Training
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    EPOCHS = int(os.getenv("EPOCHS", 100))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
    EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Data classes (5 severity levels in APTOS 2019)
    DISEASE_CLASSES = {
        0: "No DR (Diabetic Retinopathy)",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "PDR (Proliferative DR)"
    }
    NUM_CLASSES = len(DISEASE_CLASSES)
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 4))
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./prediction_logs.db")
    
    # Model architectures to train
    MODELS_TO_TRAIN = [
        "baseline_cnn",
        "custom_cnn",
        "resnet50",
        "efficientnet",
        "inceptionv3"
    ]
    
    @classmethod
    def create_all_directories(cls):
        """Create all required directories"""
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
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'image_size': cls.IMAGE_SIZE,
            'num_classes': cls.NUM_CLASSES,
            'validation_split': cls.VALIDATION_SPLIT,
            'test_split': cls.TEST_SPLIT,
        }


if __name__ == "__main__":
    Config.create_all_directories()
    print("Configuration loaded successfully!")
    print(f"Base directory: {Config.BASE_DIR}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Models directory: {Config.MODELS_DIR}")
