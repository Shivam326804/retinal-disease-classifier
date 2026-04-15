import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import Config


# ---------------------------------------------------
# Load Data (FIXED)
# ---------------------------------------------------
def load_data():
    images = np.load(Config.PROCESSED_IMAGES)
    labels = np.load(Config.PROCESSED_LABELS)

    # ❌ REMOVE normalization
    # images = images / 255.0  ← REMOVE THIS

    print(f"Loaded images: {images.shape}")
    print(f"Loaded labels: {labels.shape}")

    return images, labels


# ---------------------------------------------------
# Build Model (STABLE VERSION)
# ---------------------------------------------------
def build_model(num_classes):

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
    )

    base_model._name = "efficientnet_base"
    base_model.trainable = False

    inputs = keras.Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))

    x = data_augmentation(inputs)

    # ✅ ONLY preprocessing (no manual scaling)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=3e-4,
            clipnorm=1.0   # ✅ stabilizes training
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
def train():

    Config.set_seed(Config.RANDOM_SEED)
    Config.create_all_directories()

    images, labels = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=Config.VALIDATION_SPLIT,
        stratify=labels,
        random_state=42
    )

    num_classes = len(np.unique(labels))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))

    print("Class weights:", class_weights)

    model = build_model(num_classes)
    model.summary()

    checkpoint_path = str(Config.CHECKPOINTS_DIR / "best_model.keras")
    final_model_path = str(Config.CHECKPOINTS_DIR / "final_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ---------------------------------------------------
    # Phase 1
    # ---------------------------------------------------
    print("\n🚀 Phase 1 Training...\n")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=Config.BATCH_SIZE,
        class_weight=class_weights,
        shuffle=True,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # Load best model
    # ---------------------------------------------------
    print("\n📥 Loading best checkpoint...\n")
    model = keras.models.load_model(checkpoint_path)

    base_model = model.get_layer("efficientnet_base")

    # ---------------------------------------------------
    # Phase 2 (SAFE FINE-TUNING)
    # ---------------------------------------------------
    print("\n🔧 Fine-tuning...\n")

    # ❗ Only unfreeze TOP 30 layers (not 100)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5,
            clipnorm=1.0
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=Config.BATCH_SIZE,
        class_weight=class_weights,
        shuffle=True,
        callbacks=callbacks
    )

    model.save(final_model_path)

    print(f"\n✅ Final model saved at: {final_model_path}")


# ---------------------------------------------------
if __name__ == "__main__":
    train()