# keras_neural_network.py
# ------------------------------------------------------------
# Week 7 - Task 7.2: Neural Network with TensorFlow/Keras
# Dataset: XOR (keeps comparison easy vs MLP from scratch)
#
# What this script does:
# - Builds a small MLP with Keras Sequential API
# - Uses callbacks: EarlyStopping + ModelCheckpoint
# - Plots training history (loss + accuracy)
# - Saves model in:
#   1) H5 format (.h5)
#   2) SavedModel folder format
# - Loads saved models and tests predictions
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Reproducibility
# -----------------------------
# Keeps results consistent across runs (as much as possible)
np.random.seed(42)
tf.random.set_seed(42)


# -----------------------------
# Paths / folders
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "visuals")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# STEP 1) Load and preprocess dataset
# ============================================================
# XOR dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

# For XOR, scaling isn't needed (already 0/1),
# but in real datasets you'd normalize/standardize here.


# ============================================================
# STEP 2) Create model (Sequential API) + try architecture choices
# ============================================================
# Baseline architecture:
# - Dense hidden layer with ReLU (non-linear)
# - Output layer with sigmoid (binary probability)
model = keras.Sequential(
    [
        layers.Input(shape=(2,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="xor_mlp_keras"
)

model.summary()


# ============================================================
# STEP 3) Compile model
# ============================================================
# - optimizer: Adam is a strong default
# - loss: binary crossentropy for binary classification
# - metric: accuracy (simple and readable for XOR)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.03),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# STEP 4) Callbacks (EarlyStopping + ModelCheckpoint)
# ============================================================
# EarlyStopping: stops training when val_loss stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=60,
    restore_best_weights=True
)

# ModelCheckpoint: saves best model during training
best_h5_path = os.path.join(SAVE_DIR, "best_model.h5")
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=best_h5_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)


# ============================================================
# STEP 5) Train model (with validation_split)
# ============================================================
# XOR is tiny (4 samples), so validation_split is not very "statistically meaningful",
# but we still use it because task requires val curves + callbacks practice.
history = model.fit(
    X, y,
    epochs=500,
    batch_size=4,
    validation_split=0.25,   # uses 1 sample for validation (tiny but ok for demo)
    callbacks=[early_stop, checkpoint],
    verbose=0
)

print("\nTraining finished.")
print(f"Epochs ran: {len(history.history['loss'])}")


# ============================================================
# STEP 6) Plot training history (loss + accuracy)
# ============================================================
def plot_history(hist):
    # Loss plot
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title("Loss (Binary Crossentropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "loss_curve_keras.png"), dpi=300)
    plt.show()
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "accuracy_curve_keras.png"), dpi=300)
    plt.show()
    plt.close()

plot_history(history)


# ============================================================
# STEP 7) Save model in .h5 and SavedModel formats
# ============================================================

final_h5_path = os.path.join(SAVE_DIR, "final_model.h5")
final_keras_path = os.path.join(SAVE_DIR, "final_model.keras")
savedmodel_path = os.path.join(SAVE_DIR, "final_savedmodel")

model.save(final_h5_path)         # legacy, ok for requirement
model.save(final_keras_path)      # recommended
model.export(savedmodel_path)     # SavedModel for TFLite/Serving

print("\nModels saved successfully.")
print("H5:", final_h5_path)
print("KERAS:", final_keras_path)
print("SavedModel:", savedmodel_path)


# ============================================================
# STEP 8) Load and test saved models
# ============================================================
def predict_and_print(m, name: str):
    probs = m.predict(X, verbose=0)
    preds = (probs >= 0.5).astype(int)

    print(f"\n{name} predictions:")
    print("probs:\n", np.round(probs, 3))
    print("classes:", preds.ravel().tolist())

# Load H5 best model (saved by checkpoint)
best_model = keras.models.load_model(best_h5_path)
predict_and_print(best_model, "BEST (.h5 checkpoint)")

# Load final H5
final_h5_model = keras.models.load_model(final_h5_path)
predict_and_print(final_h5_model, "FINAL (.h5)")

# Load SavedModel
final_saved_model = keras.models.load_model(savedmodel_path)
predict_and_print(final_saved_model, "FINAL (SavedModel)")