import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.config import Config


class DataPreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        print(f"INFO: Initialized DataPreprocessor (image_size={image_size})")

    # ---------------------------------------------------
    # CLAHE Enhancement
    # ---------------------------------------------------
    def enhance_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return enhanced

    # ---------------------------------------------------
    # Load Image
    # ---------------------------------------------------
    def load_image(self, image_path):
        try:
            img = cv2.imread(str(image_path))

            if img is None:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize first (fast)
            img = cv2.resize(img, (self.image_size, self.image_size))

            # Remove invalid/dark images
            if np.mean(img) < 10:
                return None

            # Enhance
            img = self.enhance_image(img)

            return img.astype(np.float32)

        except Exception:
            return None

    # ---------------------------------------------------
    # Dataset Processing
    # ---------------------------------------------------
    def preprocess_dataset(self):
        images_dir = Path(Config.IMAGES_DIR)
        labels_path = Path(Config.LABELS_FILE)

        if not images_dir.exists():
            raise FileNotFoundError(f"❌ Images directory not found: {images_dir}")

        if not labels_path.exists():
            raise FileNotFoundError(f"❌ CSV file not found: {labels_path}")

        df = pd.read_csv(labels_path)

        images = []
        labels = []

        print("INFO: Starting preprocessing...")

        for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):

            img_name = f"{row.id_code}.png"
            label = row.diagnosis

            img_path = images_dir / img_name

            if not img_path.exists():
                continue

            img = self.load_image(img_path)

            if img is None:
                continue

            images.append(img)
            labels.append(label)

            if i % 500 == 0:
                print(f"Processed {i} images...")

        if len(images) == 0:
            raise ValueError("❌ No images processed. Check dataset path.")

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"INFO: Loaded {len(images)} images")

        return images, labels

    # ---------------------------------------------------
    # Save Data
    # ---------------------------------------------------
    def save_data(self, images, labels):
        save_dir = Path(Config.PROCESSED_DATA_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(Config.PROCESSED_IMAGES, images)
        np.save(Config.PROCESSED_LABELS, labels)

        print(f"INFO: Saved images → {Config.PROCESSED_IMAGES}")
        print(f"INFO: Saved labels → {Config.PROCESSED_LABELS}")

    # ---------------------------------------------------
    # Run Pipeline
    # ---------------------------------------------------
    def run(self):
        images, labels = self.preprocess_dataset()
        self.save_data(images, labels)

        print(f"✅ Preprocessing completed: {images.shape}")


# ---------------------------------------------------
# Entry
# ---------------------------------------------------
def main():
    preprocessor = DataPreprocessor(image_size=Config.IMAGE_SIZE)
    preprocessor.run()


if __name__ == "__main__":
    main()