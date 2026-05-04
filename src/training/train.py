"""
FINAL TRAIN.PY (OPTIMIZED + DEPLOYMENT READY)

✔ Stable pipeline
✔ Better fine-tuning strategy
✔ Model checkpoint added
✔ Improved augmentation
✔ Deployment-safe saving
"""

# ---------------------------------------------------
# SYSTEM
# ---------------------------------------------------
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.config import Config
from src.training.model_builder import ModelBuilder


# ---------------------------------------------------
# CPU SETTINGS
# ---------------------------------------------------
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

Config.set_seed(Config.RANDOM_SEED)


# ---------------------------------------------------
# 🔥 FOCAL LOSS
# ---------------------------------------------------
def focal_loss(gamma=2.0):

    class_weights = tf.constant([0.6, 1.8, 0.9, 3.0, 2.5], dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)

        weights = tf.gather(class_weights, y_true)
        focal = weights * ((1 - pt) ** gamma) * ce

        return tf.reduce_mean(focal)

    return loss


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
def load_data():
    X = np.load(Config.IMAGES_PATH)
    y = np.load(Config.LABELS_PATH)

    print(f"Images: {X.shape}")
    print(f"Labels: {y.shape}")

    return X, y


# ---------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------
def get_class_weights():
    weights = {
        0: 0.6,
        1: 1.8,
        2: 0.9,
        3: 3.0,
        4: 2.5
    }
    print("Class Weights:", weights)
    return weights


# ---------------------------------------------------
# DATA PIPELINE (IMPROVED)
# ---------------------------------------------------
def create_dataset(X, y, training=True):

    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        x = tf.image.resize(x, (260, 260))

        if training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, 0.1)

        return x, y

    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(2048)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(8)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
def train():

    print("\n🚀 Starting training...\n")

    Config.create_directories()

    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    train_ds = create_dataset(X_train, y_train, True)
    val_ds = create_dataset(X_val, y_val, False)

    class_weights = get_class_weights()

    builder = ModelBuilder(
        input_shape=(260, 260, 3),
        num_classes=Config.NUM_CLASSES
    )

    model = builder.build_model()

    # ---------------------------------------------------
    # CALLBACKS (IMPROVED)
    # ---------------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODELS_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )
    ]

    # -------------------------
    # STAGE 1
    # -------------------------
    print("\n📌 Stage 1: Training\n")

    model = builder.compile_model(
        model,
        lr=1e-4,
        loss_fn=focal_loss()
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # -------------------------
    # FINE TUNE (FIXED)
    # -------------------------
    print("\n🔥 Fine-tuning...\n")

    model = builder.fine_tune_model(model, 120)   # 🔥 reduced from 180

    model = builder.compile_model(
        model,
        lr=1e-5   # 🔥 increased from 2e-6
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks,
        verbose=2
    )

    # ---------------------------------------------------
    # SAVE FINAL MODEL
    # ---------------------------------------------------
    save_path = os.path.join(Config.MODELS_DIR, "final_model.keras")

    model.save(save_path)

    print(f"\n✅ Model saved at: {save_path}")
    print("🎉 Training complete!")


if __name__ == "__main__":
    train()