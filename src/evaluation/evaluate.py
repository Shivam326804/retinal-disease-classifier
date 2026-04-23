import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
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
# 🔥 CALIBRATION (IMPROVED)
# ---------------------------------------------------
def calibrate_predictions(preds):
    """
    Slightly boost severe classes without over-distortion
    """

    calibrated = preds.copy()

    # softer scaling (more stable)
    weights = np.array([1.0, 1.0, 1.05, 1.20, 1.15])

    calibrated = calibrated * weights

    # safe normalization
    calibrated = calibrated / (np.sum(calibrated, axis=1, keepdims=True) + 1e-8)

    return calibrated


# ---------------------------------------------------
# 🔥 TTA PREDICTION (OPTIMIZED)
# ---------------------------------------------------
def tta_predict(model, X, batch_size=16):

    print("🔁 Running TTA predictions...")

    all_preds = []

    for i in range(0, len(X), batch_size):

        batch = X[i:i + batch_size]

        tta_images = []

        for img in batch:

            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, (260, 260))

            # augmentations
            variants = [
                img,
                tf.image.flip_left_right(img),
                tf.image.flip_up_down(img)
            ]

            variants = [
                tf.keras.applications.efficientnet.preprocess_input(v)
                for v in variants
            ]

            tta_images.extend(variants)

        tta_images = tf.stack(tta_images)

        preds = model.predict(tta_images, verbose=0)

        # reshape: (batch, 3, num_classes)
        preds = preds.reshape(len(batch), 3, -1)

        preds = np.mean(preds, axis=1)

        all_preds.append(preds)

    preds = np.vstack(all_preds)

    # 🔥 calibration
    preds = calibrate_predictions(preds)

    return preds


# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------
def evaluate():

    print("\n📊 Running evaluation with TTA + Calibration...\n")

    # ensure dirs exist
    Config.create_directories()

    X, y = load_data()

    print("🔄 Loading model...")
    model = tf.keras.models.load_model(
        Config.get_model_path(),
        compile=False
    )
    print("✅ Model loaded")

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------
    preds = tta_predict(model, X)
    y_pred = np.argmax(preds, axis=1)

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------
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
    plt.title("Normalized Confusion Matrix (TTA + Calibrated)")

    save_path = Config.REPORTS_DIR / "confusion_matrix_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"📁 Confusion matrix saved at: {save_path}")

    # ---------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------
    report = classification_report(
        y,
        y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4
    )

    print("\n📄 Classification Report (Final):\n")
    print(report)

    report_path = Config.REPORTS_DIR / "classification_report_final.txt"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"📁 Report saved at: {report_path}")

    # ---------------------------------------------------
    # CLASS DISTRIBUTION
    # ---------------------------------------------------
    print("\n📊 Class distribution:")
    print(dict(zip(Config.CLASS_NAMES, np.bincount(y))))

    print("\n✅ Evaluation complete!")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    evaluate()