"""
Prediction Script
Makes predictions on new images
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.inference import Predictor, GradCAMVisualizer
from src.utils.data_utils import get_image_files, save_image

logger = setup_logger(__name__)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_predictor(model_name: Optional[str] = None) -> Optional[Predictor]:

    if model_name is None:
        model_name = Config.MODEL_NAME

    model_path = Path(Config.MODELS_DIR) / model_name

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None

    predictor = Predictor(str(model_path), Config.DISEASE_CLASSES)

    if predictor.model is None:
        logger.error("Model failed to load.")
        return None

    return predictor


# ---------------------------------------------------
# SINGLE IMAGE PREDICTION
# ---------------------------------------------------

def predict_single_image(
    image_path: str,
    model_name: Optional[str] = None,
    generate_gradcam: bool = False
) -> Optional[Dict[str, Any]]:

    logger.info(f"\nPredicting image: {image_path}")

    predictor = load_predictor(model_name)

    if predictor is None or predictor.model is None:
        return None

    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)

    image = preprocessor.load_image(
        Path(image_path),
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    )

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    predicted_class, confidence, probabilities = predictor.predict(image)

    prob_dist = predictor.get_prediction_confidence_distribution(probabilities)

    logger.info(f"Prediction: {predicted_class}")
    logger.info(f"Confidence: {confidence:.4f}")

    logger.info("\nProbability Distribution:")

    for disease, prob in sorted(prob_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{disease}: {prob:.4f}")

    # ---------------------------------------------------
    # GRADCAM
    # ---------------------------------------------------

    gradcam_path = None

    if generate_gradcam:

        try:

            logger.info("Generating Grad-CAM visualization...")

            # FIX: Explicit check for model
            model = predictor.model
            if model is None:
                raise ValueError("Predictor model is None")

            visualizer = GradCAMVisualizer(model)

            class_idx = int(np.argmax(probabilities))

            heatmap = visualizer.generate_cam(image, class_idx)

            visualization = visualizer.visualize_with_heatmap(image, heatmap)

            gradcam_path = str(Path(image_path).with_name(
                Path(image_path).stem + "_gradcam.png"
            ))

            save_image(visualization, gradcam_path)

            logger.info(f"Grad-CAM saved: {gradcam_path}")

        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")

    return {
        "image_path": image_path,
        "prediction": predicted_class,
        "confidence": float(confidence),
        "probabilities": prob_dist,
        "gradcam": gradcam_path,
    }


# ---------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------

def predict_batch(
    image_dir: str,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:

    logger.info(f"Running batch prediction on: {image_dir}")

    image_files = get_image_files(image_dir)

    if not image_files:
        logger.warning("No images found in directory.")
        return []

    predictor = load_predictor(model_name)

    if predictor is None or predictor.model is None:
        return []

    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)

    results: List[Dict[str, Any]] = []

    for image_path in image_files:

        image = preprocessor.load_image(
            Path(image_path),
            target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        )

        if image is None:
            continue

        predicted_class, confidence, probabilities = predictor.predict(image)

        results.append(
            {
                "image": image_path,
                "prediction": predicted_class,
                "confidence": float(confidence),
            }
        )

    logger.info(f"Batch prediction completed: {len(results)} images")

    return results


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Retinal disease prediction")

    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image"
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory of images"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model filename inside models folder"
    )

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM"
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.print_help()
        sys.exit(1)

    # Single image
    if args.image:

        predict_single_image(
            args.image,
            args.model,
            args.gradcam
        )

    # Batch prediction
    if args.image_dir:

        results = predict_batch(
            args.image_dir,
            args.model
        )

        if results:

            logger.info("\nBatch Prediction Summary\n")

            for r in results[:10]:

                logger.info(
                    f"{r['image']} -> "
                    f"{r['prediction']} "
                    f"({r['confidence']:.4f})"
                )

            if len(results) > 10:

                logger.info(
                    f"... and {len(results) - 10} more images"
                )


if __name__ == "__main__":
    main()