
# ==================================================================
# Task 7.4 — Transfer Learning with Pre-trained Models (MobileNetV2)
# ==================================================================


from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# =========================
# 0) Repro + folders
# =========================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = BASE_DIR / "visuals"
MODELS_DIR = BASE_DIR / "saved_models"
VIS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Dataset (Cats vs Dogs)
# =========================
# We’ll use TFDS so you don't manually download zip files.
# cats_vs_dogs dataset usually has ~23k images.
DATASET_NAME = "tf_flowers"

IMG_SIZE = (160, 160)   # MobileNetV2 loves 160/224-ish sizes
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def preprocess_example(example: dict) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Converts raw TFDS example to (image, label)
    - Resize
    - Convert dtype
    """
    image = example["image"]
    label = example["label"]

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)  # keep float32 for model input
    return image, label


def make_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Train/Val split. TFDS supports slicing with percentages.
    We'll use 80/20 split.
    """
    train_ds = tfds.load(DATASET_NAME, split="train[:80%]", shuffle_files=True)
    val_ds = tfds.load(DATASET_NAME, split="train[80%:]", shuffle_files=False)

    train_ds = train_ds.map(preprocess_example, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_example, num_parallel_calls=AUTOTUNE)

    # Shuffle only train
    train_ds = train_ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)

    # Batch + prefetch
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds


train_ds, val_ds = make_datasets()


# =========================
# 2) Data Augmentation
# =========================
# Augmentation improves generalization.
# Keep it light + realistic for cats/dogs.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.08, seed=SEED),
        tf.keras.layers.RandomZoom(0.10, seed=SEED),
        tf.keras.layers.RandomContrast(0.10, seed=SEED),
    ],
    name="augmentation",
)


# =========================
# 3) Utility: training helper
# =========================
def compile_and_train(model: tf.keras.Model,
                      train_data: tf.data.Dataset,
                      val_data: tf.data.Dataset,
                      epochs: int,
                      lr: float,
                      tag: str) -> dict:
    """
    Trains a model and returns key metrics.
    'tag' used for saving history plots nicely.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    t0 = time.time()
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks, verbose=1)
    t1 = time.time()

    # Get best val accuracy achieved in history
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_val_loss = float(np.min(history.history["val_loss"]))

    # Simple plot: accuracy curves (optional, but nice for submission)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title(f"Accuracy vs Epochs — {tag}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"acc_{tag}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title(f"Loss vs Epochs — {tag}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"loss_{tag}.png", dpi=300)
        plt.close()
    except Exception:
        # If matplotlib missing, training still fine. (But you already have it.)
        pass

    return {
        "tag": tag,
        "epochs_ran": len(history.history["loss"]),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "train_time_sec": round(t1 - t0, 2),
    }


# =========================
# 4) Model A — Transfer Learning (MobileNetV2)
# =========================
NUM_CLASSES = 5

# MobileNetV2 preprocessing: expects inputs in [-1, 1]
mobilenet_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Step 4: freeze base layers
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")

# augmentation only during training (Keras handles automatically)
x = data_augmentation(inputs)

# Normalize for MobileNetV2
x = mobilenet_preprocess(x)

# Extract features
x = base_model(x, training=False)

# Pool features (turn HxWxC into 1x1xC -> C)
x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

# Regularization to reduce overfit on small datasets
x = tf.keras.layers.Dropout(0.25, name="drop")(x)

# Classification head
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)

transfer_model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")


# =========================
# 5) Train head (base frozen)
# =========================
transfer_stage1 = compile_and_train(
    model=transfer_model,
    train_data=train_ds,
    val_data=val_ds,
    epochs=8,
    lr=1e-3,
    tag="transfer_frozen"
)


# =========================
# 6) Fine-tuning (unfreeze some layers)
# =========================
# Step 7: unfreeze a few deeper layers
# Idea: keep early layers frozen (generic edges/textures),
# allow later layers to adapt to cats/dogs.
base_model.trainable = True

# Freeze first N layers, unfreeze rest
fine_tune_at = 100  # tweak if needed
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

transfer_stage2 = compile_and_train(
    model=transfer_model,
    train_data=train_ds,
    val_data=val_ds,
    epochs=6,
    lr=1e-4,      # lower LR for fine-tuning (important)
    tag="transfer_finetune"
)


# =========================
# 7) Model B — Train from Scratch (baseline CNN)
# =========================
# This is intentionally simple. It's our "control experiment".
def build_scratch_cnn() -> tf.keras.Model:
    inp = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = data_augmentation(inp)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inp, out, name="scratch_cnn")


scratch_model = build_scratch_cnn()

scratch_result = compile_and_train(
    model=scratch_model,
    train_data=train_ds,
    val_data=val_ds,
    epochs=10,
    lr=1e-3,
    tag="scratch_cnn"
)


# =========================
# 8) Performance Comparison Table
# =========================
def print_comparison_table(rows: list[dict]) -> None:
    # tiny table printer (no pandas dependency)
    headers = ["Model", "Epochs", "Best Val Acc", "Best Val Loss", "Train Time (s)"]
    widths = [24, 8, 14, 14, 15]

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(fmt_row(headers))
    print("-" * 80)
    for r in rows:
        print(fmt_row([
            r["tag"],
            r["epochs_ran"],
            f"{r['best_val_acc']:.4f}",
            f"{r['best_val_loss']:.4f}",
            r["train_time_sec"]
        ]))
    print("=" * 80 + "\n")


comparison_rows = [transfer_stage1, transfer_stage2, scratch_result]
print_comparison_table(comparison_rows)

# Save comparison as CSV (easy for report)
csv_path = VIS_DIR / "performance_comparison.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("model,epochs_ran,best_val_acc,best_val_loss,train_time_sec\n")
    for r in comparison_rows:
        f.write(f"{r['tag']},{r['epochs_ran']},{r['best_val_acc']:.6f},{r['best_val_loss']:.6f},{r['train_time_sec']}\n")

print(f"Saved comparison CSV -> {csv_path}")


# =========================
# 9) Save final (Transfer) model in multiple formats
# =========================
# We'll save the fine-tuned transfer_model as the "final" one.

# (a) H5 (legacy but commonly requested)
h5_path = MODELS_DIR / "final_transfer_model.h5"
transfer_model.save(h5_path)
print(f"Saved H5 model -> {h5_path}")

# (b) Native Keras format (recommended these days)
keras_path = MODELS_DIR / "final_transfer_model.keras"
transfer_model.save(keras_path)
print(f"Saved Keras model -> {keras_path}")

# (c) SavedModel export
# In Keras 3, use model.export() for SavedModel-style directory.
savedmodel_dir = MODELS_DIR / "final_transfer_savedmodel"
transfer_model.export(savedmodel_dir)
print(f"Exported SavedModel -> {savedmodel_dir}")

# (d) TFLite conversion
# Convert from the Keras model directly
converter = tf.lite.TFLiteConverter.from_keras_model(transfer_model)

# optional: size optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
tflite_path = MODELS_DIR / "final_transfer_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model -> {tflite_path}")

print("\n✅ Task 7.4 done: transfer learning + fine-tune + scratch compare + saved models.\n")