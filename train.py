"""
Training Script
Trains all model architectures on the prepared dataset
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.dataset_loader import DatasetLoader
from src.training.model_builder import ModelBuilder
from src.training.train import Trainer

logger = setup_logger(__name__)


def load_and_split_data():
    """Load preprocessed data and split into train/val/test"""
    logger.info("Loading preprocessed data...")
    
    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)
    images, labels = preprocessor.load_processed_data()
    
    if images is None or labels is None:
        logger.error("Preprocessed data not found. Run download_dataset.py first.")
        sys.exit(1)
    
    # Split dataset using original labels (1D) for stratification
    splits = preprocessor.split_dataset(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Convert labels in training and validation splits to one-hot encoding
    for key in ['train', 'val']:
        X, y = splits[key]
        y_onehot = np.eye(Config.NUM_CLASSES)[y]
        splits[key] = (X, y_onehot)
    
    return splits


def train_model(model_name, train_dataset, val_dataset, epochs, class_weights=None):
    """Train a single model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model: {model_name}")
    logger.info(f"{'='*60}")
    
    # Build model
    builder = ModelBuilder(
        input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
        num_classes=Config.NUM_CLASSES
    )
    
    if model_name == "baseline_cnn":
        model = builder.build_baseline_cnn()
    elif model_name == "custom_cnn":
        model = builder.build_custom_cnn()
    elif model_name == "resnet50":
        model = builder.build_resnet50()
    elif model_name == "efficientnet":
        model = builder.build_efficientnet()
    elif model_name == "inceptionv3":
        model = builder.build_inceptionv3()
    else:
        logger.error(f"Unknown model: {model_name}")
        return None
    
    # Print model summary
    logger.info("\nModel Architecture:")
    logger.info(builder.get_model_summary(model))
    
    # Train model
    trainer = Trainer(model_name=model_name)
    history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        class_weights=class_weights
    )
    
    # Save model
    model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.keras")
    trainer.save_model(model, model_path)
    
    # Save training history
    trainer.save_training_history(
        os.path.join(Config.LOGS_DIR, f"{model_name}_history.json")
    )
    
    logger.info(f"Model saved to {model_path}")
    
    return model, history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train retinal disease classification models")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--models", type=str, default="baseline_cnn,custom_cnn",
                       help="Comma-separated list of models to train")
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_all_directories()
    
    logger.info(f"Starting training pipeline...")
    logger.info(f"Configuration: {Config.get_config_dict()}")
    
    # Load and split data
    splits = load_and_split_data()
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Compute class weights
    y_train_1d = np.argmax(y_train, axis=1)
    class_weights = DatasetLoader.get_class_weights(y_train_1d)
    
    # Create data generators
    train_dataset, val_dataset = DatasetLoader.create_generators(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size
    )
    
    # Parse models to train
    models_to_train = [m.strip() for m in args.models.split(',')]
    
    # Train models
    training_results = {}
    for model_name in models_to_train:
        try:
            result = train_model(
                model_name=model_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
                class_weights=class_weights
            )
            if result:
                model, history = result
                training_results[model_name] = {
                    'status': 'completed',
                    'epochs': len(history['loss']),
                    'final_train_loss': float(history['loss'][-1]),
                    'final_val_loss': float(history['val_loss'][-1]),
                    'best_val_accuracy': float(max(history['val_accuracy']))
                }
                logger.info(f"✓ {model_name} training completed")
        except Exception as e:
            logger.error(f"✗ {model_name} training failed: {str(e)}")
            training_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save training results
    results_path = os.path.join(Config.LOGS_DIR, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=4)
    
    logger.info(f"\n{'='*60}")
    logger.info("Training Results Summary")
    logger.info(f"{'='*60}")
    for model_name, result in training_results.items():
        logger.info(f"\n{model_name}:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
    
    logger.info(f"\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
