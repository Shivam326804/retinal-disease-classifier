import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.metrics import confusion_matrix, classification_report
from src.utils.config import Config


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
def load_data():
    print("📂 Loading dataset...")

    X = np.load(Config.IMAGES_PATH)
    y = np.load(Config.LABELS_PATH)

    print(f"Images: {X.shape}")
    print(f"Labels: {y.shape}")

    return X, y


# ---------------------------------------------------
# PREPROCESS (MATCH TRAINING)
# ---------------------------------------------------
def crop_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    if w < 50 or h < 50:
        return img

    return img[y:y+h, x:x+w]


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(img):
    img = crop_retina(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (260, 260))
    return img.astype("float32")


# ---------------------------------------------------
# CALIBRATION
# ---------------------------------------------------
def calibrate_predictions(preds):

    weights = np.array([1.0, 1.0, 1.05, 1.20, 1.15])
    preds = preds * weights
    preds = preds / (np.sum(preds, axis=1, keepdims=True) + 1e-8)

    return preds


# ---------------------------------------------------
# TTA (FIXED)
# ---------------------------------------------------
def tta_predict(model, X, batch_size=16):

    print("🔁 Running TTA predictions...")

    all_preds = []

    for i in range(0, len(X), batch_size):

        batch = X[i:i + batch_size]
        tta_batch = []

        for img in batch:

            img = preprocess(img)

            variants = [
                img,
                cv2.flip(img, 1),
                cv2.flip(img, 0)
            ]

            tta_batch.extend(variants)

        tta_batch = np.array(tta_batch)

        preds = model.predict(tta_batch, verbose=0)

        preds = preds.reshape(len(batch), 3, -1)
        preds = np.mean(preds, axis=1)

        all_preds.append(preds)

    preds = np.vstack(all_preds)

    preds = calibrate_predictions(preds)

    return preds


# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------
def evaluate():

    print("\n📊 Running evaluation (FIXED PIPELINE)...\n")

    Config.create_directories()

    X, y = load_data()

    print("🔄 Loading model...")
    model = tf.keras.models.load_model(
        Config.get_model_path(),
        compile=False
    )
    print("✅ Model loaded")

    preds = tta_predict(model, X)
    y_pred = np.argmax(preds, axis=1)

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(y, y_pred)

    cm_norm = cm.astype("float") / (
        cm.sum(axis=1, keepdims=True) + 1e-8
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")

    save_path = Config.REPORTS_DIR / "confusion_matrix_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"📁 Saved: {save_path}")

    # -------------------------
    # REPORT
    # -------------------------
    report = classification_report(
        y,
        y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4
    )

    print("\n📄 Classification Report:\n")
    print(report)

    report_path = Config.REPORTS_DIR / "classification_report_final.txt"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"📁 Saved: {report_path}")

    print("\n✅ Evaluation complete!")


# ---------------------------------------------------
if __name__ == "__main__":
    evaluate()