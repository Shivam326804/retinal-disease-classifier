import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple


class Predictor:

    def __init__(self, model_path: str = "models/final_model.keras"):

        print("🔄 Loading model...")

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("✅ Model loaded successfully")

        self.class_names = [
            "No DR",
            "Mild NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Proliferative DR"
        ]

    # ---------------------------------------------------
    # RETINA CROP (MATCH TRAINING)
    # ---------------------------------------------------
    def crop_retina(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)

        if coords is None:
            return img

        x, y, w, h = cv2.boundingRect(coords)

        if w < 50 or h < 50:
            return img

        return img[y:y+h, x:x+w]

    # ---------------------------------------------------
    # CLAHE
    # ---------------------------------------------------
    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ---------------------------------------------------
    # PREPROCESS (FIXED)
    # ---------------------------------------------------
    def preprocess(self, image):

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("❌ Invalid image")

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 Match training pipeline
        image = self.crop_retina(image)
        image = self.apply_clahe(image)
        image = cv2.resize(image, (260, 260))

        # ❌ DO NOT preprocess_input here
        image = image.astype("float32")

        return image

    # ---------------------------------------------------
    # TTA
    # ---------------------------------------------------
    def tta_augment(self, image):

        variants = [image]

        variants.append(cv2.flip(image, 1))

        h, w = image.shape[:2]

        for angle in [8, -8]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (w, h))
            variants.append(rotated)

        variants.append(cv2.convertScaleAbs(image, alpha=1.1, beta=10))
        variants.append(cv2.convertScaleAbs(image, alpha=0.9, beta=-10))

        return variants

    # ---------------------------------------------------
    # CALIBRATION
    # ---------------------------------------------------
    def calibrate(self, probs):

        calibrated = probs.copy()

        calibrated[3] *= 1.25
        calibrated[4] *= 1.20

        calibrated = calibrated / np.sum(calibrated)

        return calibrated

    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------
    def predict(self, image) -> Tuple[str, float, np.ndarray]:

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("❌ Invalid image")

        # Preprocess once
        base = self.preprocess(image)

        # Apply TTA on processed image
        variants = self.tta_augment(base)

        batch = np.array(variants)

        preds = self.model.predict(batch, verbose=0)

        weights = np.array([1.0] + [0.9] * (len(preds) - 1))
        probs = np.average(preds, axis=0, weights=weights)

        probs = self.calibrate(probs)

        class_id = int(np.argmax(probs))
        confidence = float(np.max(probs))

        label = self.class_names[class_id]

        if confidence > 0.75:
            conf_text = "High Confidence"
        elif confidence > 0.50:
            conf_text = "Moderate Confidence"
        else:
            conf_text = "Low Confidence"

        final_label = f"{label} ({conf_text})"

        return final_label, confidence, probs

    # ---------------------------------------------------
    def predict_single(self, image):

        image = self.preprocess(image)
        image = np.expand_dims(image, axis=0)

        probs = self.model.predict(image, verbose=0)[0]
        probs = self.calibrate(probs)

        return probs

    def get_model(self):
        return self.model

    def get_classes(self):
        return self.class_names