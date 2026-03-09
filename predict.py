"""
Prediction Script
Makes predictions on new images
"""

import os
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

logger = setup_logger(__name__)


def predict_single_image(
    image_path: str,
    model_name: str = "baseline_cnn",
    generate_gradcam: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Predict disease class for a single image
    """

    logger.info(f"\nPredicting on image: {image_path}")

    # Load model
    model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.h5")
    predictor = Predictor(model_path, Config.DISEASE_CLASSES)

    if predictor.model is None:
        logger.error("Model failed to load.")
        return None

    # Preprocessor
    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)

    # FIX: convert to Path
    image = preprocessor.load_image(
        Path(image_path),
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    )

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    # Predict
    predicted_class, confidence, probabilities = predictor.predict(image)

    logger.info(f"Predicted Disease: {predicted_class}")
    logger.info(f"Confidence: {confidence:.4f}")

    logger.info("\nConfidence Distribution:")

    prob_dist = predictor.get_prediction_confidence_distribution(probabilities)

    for disease, prob in sorted(prob_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {disease}: {prob:.4f}")

    # Grad-CAM
    if generate_gradcam:
        try:

            logger.info("\nGenerating Grad-CAM visualization...")

            visualizer = GradCAMVisualizer(predictor.model)

            class_idx = int(np.argmax(probabilities))

            heatmap = visualizer.generate_cam(image, class_idx)

            visualization = visualizer.visualize_with_heatmap(image, heatmap)

            output_path = Path(image_path).stem + "_gradcam.png"

            from src.utils.data_utils import save_image

            save_image(visualization, output_path)

            logger.info(f"Grad-CAM visualization saved to {output_path}")

        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {str(e)}")

    return {
        "image": image_path,
        "predicted_disease": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dist,
    }


def predict_batch(
    image_dir: str,
    model_name: str = "baseline_cnn"
) -> List[Dict[str, Any]]:
    """
    Predict on all images in a directory
    """

    logger.info(f"Batch prediction on directory: {image_dir}")

    from src.utils.data_utils import get_image_files

    image_files = get_image_files(image_dir)

    if not image_files:
        logger.warning("No images found in directory")
        return []

    model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.h5")

    predictor = Predictor(model_path, Config.DISEASE_CLASSES)

    if predictor.model is None:
        logger.error("Model failed to load.")
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

        predicted_class, confidence, _ = predictor.predict(image)

        results.append(
            {
                "image": image_path,
                "predicted_disease": predicted_class,
                "confidence": confidence,
            }
        )

    logger.info(f"Completed batch prediction: {len(results)} images")

    return results


def main() -> None:
    """Main prediction function"""

    parser = argparse.ArgumentParser(description="Make predictions on images")

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for prediction",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images for batch prediction",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="baseline_cnn",
        help="Name of trained model to use",
    )

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM visualization",
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.print_help()
        sys.exit(1)

    # Single prediction
    if args.image:

        predict_single_image(
            args.image,
            args.model,
            args.gradcam,
        )

    # Batch prediction
    if args.image_dir:

        results = predict_batch(
            args.image_dir,
            args.model,
        )

        if results:

            logger.info("\n" + "=" * 60)
            logger.info("Batch Prediction Summary")
            logger.info("=" * 60)

            for result in results[:10]:

                logger.info(
                    f"{result['image']}: "
                    f"{result['predicted_disease']} "
                    f"({result['confidence']:.4f})"
                )

            if len(results) > 10:

                logger.info(
                    f"... and {len(results) - 10} more predictions"
                )


if __name__ == "__main__":
    main()