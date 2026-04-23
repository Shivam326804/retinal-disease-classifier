"""
FINAL data_preprocessor.py (STABLE + FAST + DEBUG SAFE)

✔ Correct config usage
✔ Robust dataset handling
✔ Safe retina crop
✔ CLAHE preserved
✔ Debug-friendly logs
✔ No silent failures
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.config import Config


class DataPreprocessor:

    def __init__(self, image_size=224):
        self.image_size = image_size
        print(f"INFO: Preprocessor initialized ({image_size}x{image_size})")

    # ---------------------------------------------------
    # SAFE RETINA CROP
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
    def clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ---------------------------------------------------
    # IMAGE PIPELINE
    # ---------------------------------------------------
    def process_image(self, image_path):

        img = cv2.imread(str(image_path))

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.crop_retina(img)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Reject very dark images
        if np.mean(img) < 10:
            return None

        img = self.clahe(img)

        return img.astype(np.uint8)

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------
    def preprocess_dataset(self):

        images_dir = Path(Config.IMAGES_DIR)
        labels_csv = Path(Config.LABELS_FILE)

        if not images_dir.exists():
            raise FileNotFoundError(f"❌ Images folder not found: {images_dir}")

        if not labels_csv.exists():
            raise FileNotFoundError(f"❌ Labels file not found: {labels_csv}")

        df = pd.read_csv(labels_csv)

        # 🔥 safety check
        if "id_code" not in df.columns or "diagnosis" not in df.columns:
            raise ValueError("❌ CSV must contain 'id_code' and 'diagnosis' columns")

        images = []
        labels = []

        skipped = 0

        print("🚀 Processing dataset...")

        for row in tqdm(df.itertuples(index=False), total=len(df)):

            img_path = images_dir / f"{row.id_code}.png"

            if not img_path.exists():
                skipped += 1
                continue

            img = self.process_image(img_path)

            if img is None:
                skipped += 1
                continue

            images.append(img)
            labels.append(row.diagnosis)

        if len(images) == 0:
            raise ValueError("❌ No valid images processed")

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)

        print("\n📊 DATASET SUMMARY")
        print(f"✅ Final images: {images.shape}")
        print(f"⚠️ Skipped: {skipped}")

        return images, labels

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------
    def save(self, images, labels):

        save_dir = Path(Config.DATA_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(Config.IMAGES_PATH, images)
        np.save(Config.LABELS_PATH, labels)

        print("\n💾 Dataset saved:")
        print(f" - Images: {Config.IMAGES_PATH}")
        print(f" - Labels: {Config.LABELS_PATH}")

    # ---------------------------------------------------
    # RUN
    # ---------------------------------------------------
    def run(self):

        images, labels = self.preprocess_dataset()
        self.save(images, labels)

        print("\n🎉 Preprocessing completed successfully!")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():

    Config.create_directories()

    preprocessor = DataPreprocessor(
        image_size=Config.IMAGE_SIZE
    )

    preprocessor.run()


if __name__ == "__main__":
    main()