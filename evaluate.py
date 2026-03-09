"""
Evaluation Module
Evaluates trained models on the test dataset
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.dataset_loader import DatasetLoader

logger = setup_logger(__name__)


# ------------------------------------------------------------
# Load Test Data
# ------------------------------------------------------------

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:

    logger.info("Loading processed dataset")

    processed_dir = Config.PROCESSED_DATA_DIR

    images_path = processed_dir / "images.npy"
    labels_path = processed_dir / "labels.npy"

    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run preprocessing first."
        )

    images = np.load(images_path)
    labels = np.load(labels_path)

    logger.info(f"Dataset loaded: {images.shape}")

    _, X_test, _, y_test = train_test_split(
        images,
        labels,
        test_size=Config.TEST_SPLIT,
        random_state=42,
        stratify=labels,
    )

    logger.info(f"Test samples: {len(X_test)}")

    return X_test, y_test


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

def load_model() -> tf.keras.Model:

    model_path = Config.MODELS_DIR / Config.MODEL_NAME

    if not model_path.exists():

        for ext in [".keras", ".h5"]:
            alt_path = Config.MODELS_DIR / f"retinal_disease_classifier{ext}"

            if alt_path.exists():
                model_path = alt_path
                break

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found.")

    logger.info(f"Loading model from {model_path}")

    model = tf.keras.models.load_model(model_path)

    logger.info("Model loaded successfully")

    return model


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

def evaluate_model(
    model: tf.keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Any]:

    logger.info("Starting evaluation")

    dataset_loader = DatasetLoader(batch_size=Config.BATCH_SIZE)

    test_dataset = dataset_loader.create_tf_dataset(
        test_images,
        test_labels,
        augment=False,
        shuffle=False,
    )

    results = model.evaluate(test_dataset, verbose=1)

    if isinstance(results, (list, tuple)):
        loss = float(results[0])
        accuracy = float(results[1]) if len(results) > 1 else 0.0
    else:
        loss = float(results)
        accuracy = 0.0

    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_dataset, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_labels

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    conf_matrix = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(Config.DISEASE_CLASSES.values()),
    )

    metrics = {
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
    }

    return metrics


# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------

def save_results(metrics: Dict[str, Any]) -> None:

    results_dir = Config.REPORTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = results_dir / "evaluation_metrics.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    report_file = results_dir / "classification_report.txt"

    with open(report_file, "w") as f:
        f.write(metrics["classification_report"])

    logger.info("Results saved successfully")


# ------------------------------------------------------------
# Plot Confusion Matrix
# ------------------------------------------------------------

def generate_plots(metrics: Dict[str, Any]) -> None:

    results_dir = Config.REPORTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    conf_matrix = np.array(metrics["confusion_matrix"])

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(Config.DISEASE_CLASSES.values()),
        yticklabels=list(Config.DISEASE_CLASSES.values()),
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plot_path = results_dir / "confusion_matrix.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Confusion matrix saved to {plot_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    logger.info("Starting evaluation pipeline")

    try:

        test_images, test_labels = load_test_data()

        model = load_model()

        metrics = evaluate_model(model, test_images, test_labels)

        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)

        print(f"Accuracy : {metrics['test_accuracy']:.4f}")
        print(f"Loss     : {metrics['test_loss']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")

        print("\nClassification Report:\n")
        print(metrics["classification_report"])

        save_results(metrics)

        generate_plots(metrics)

        logger.info("Evaluation completed successfully")

    except Exception as e:

        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()