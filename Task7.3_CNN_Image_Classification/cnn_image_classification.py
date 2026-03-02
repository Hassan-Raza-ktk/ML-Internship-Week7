# cnn_image_classification.py
# ------------------------------------------------------------
# Week 7 - Task 7.3: CNN for Image Classification (MNIST/Fashion-MNIST)
#
# Features included (as per task):
# 1) Load dataset (MNIST/Fashion-MNIST)
# 2) Reshape for CNN: (samples, height, width, channels)
# 3) Build CNN: Conv2D, MaxPooling, Dropout, Dense
# 4) Data augmentation (ImageDataGenerator)
# 5) Train model + plot history
# 6) Visualize first-layer filters
# 7) Visualize feature maps for a sample image
# 8) Confusion matrix evaluation
# 9) Convert model to TFLite (.tflite)
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# Reproducibility
# -----------------------------
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
# STEP 1) Load MNIST or Fashion-MNIST dataset
# ============================================================
USE_FASHION = True  # True = Fashion-MNIST, False = MNIST

if USE_FASHION:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    ds_name = "fashion_mnist"
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    class_names = [str(i) for i in range(10)]
    ds_name = "mnist"

print(f"Dataset: {ds_name}")
print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)


# ============================================================
# STEP 2) Reshape data for CNN + normalize
# ============================================================
# Original shape: (N, 28, 28)
# CNN expects:    (N, 28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

print("After reshape -> Train:", x_train.shape, "Test:", x_test.shape)


# ============================================================
# STEP 3) Build CNN architecture
# ============================================================
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(10, activation="softmax"),
    ],
    name="cnn_classifier"
)

model.summary()


# ============================================================
# STEP 4) Compile model
# ============================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# STEP 5) Data augmentation (ImageDataGenerator)
# ============================================================
# Note: MNIST/FashionMNIST are simple; augmentation is for practice
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10
)

datagen.fit(x_train)

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

best_model_path = os.path.join(SAVE_DIR, f"best_{ds_name}.keras")
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# STEP 6) Train model
# ============================================================
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\nTraining finished.")
print(f"Epochs ran: {len(history.history['loss'])}")


# ============================================================
# STEP 7) Plot training history (loss & accuracy)
# ============================================================
def plot_history(hist):
    # Loss
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title(f"Loss - {ds_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{ds_name}_loss.png"), dpi=300)
    plt.show()
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.title(f"Accuracy - {ds_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{ds_name}_accuracy.png"), dpi=300)
    plt.show()
    plt.close()

plot_history(history)


# ============================================================
# STEP 8) Visualize first layer filters
# ============================================================
def visualize_first_layer_filters(m):
    # First Conv layer weights: (kernel_h, kernel_w, in_channels, out_channels)
    conv_layer = None
    for layer in m.layers:
        if isinstance(layer, layers.Conv2D):
            conv_layer = layer
            break

    if conv_layer is None:
        print("No Conv2D layer found for filter visualization.")
        return

    weights, bias = conv_layer.get_weights()
    # weights shape: (3,3,1,32) typically
    num_filters = weights.shape[-1]

    # Normalize each filter for viewing
    filt = weights[:, :, 0, :]  # (kh, kw, out_channels)

    cols = 8
    rows = int(np.ceil(num_filters / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_filters):
        ax = plt.subplot(rows, cols, i + 1)
        f = filt[:, :, i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-9)
        ax.imshow(f, cmap="gray")
        ax.set_title(f"F{i}")
        ax.axis("off")

    plt.suptitle("First Conv Layer Filters", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{ds_name}_first_layer_filters.png"), dpi=300)
    plt.show()
    plt.close()

visualize_first_layer_filters(model)


# ============================================================
# STEP 9) Visualize feature maps for one sample image
# ============================================================
def visualize_feature_maps(m, sample_image):
    # Build a model that outputs activations of Conv layers
    conv_outputs = []
    conv_names = []
    for layer in m.layers:
        if isinstance(layer, layers.Conv2D):
            conv_outputs.append(layer.output)
            conv_names.append(layer.name)

    if not conv_outputs:
        print("No Conv2D layers found for feature maps.")
        return

    activation_model = keras.Model(inputs=m.inputs, outputs=conv_outputs)

    # Add batch dimension
    sample = np.expand_dims(sample_image, axis=0)
    activations = activation_model.predict(sample, verbose=0)

    for act, name in zip(activations, conv_names):
        # act shape: (1, h, w, channels)
        channels = act.shape[-1]
        show_n = min(16, channels)

        cols = 4
        rows = int(np.ceil(show_n / cols))

        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(show_n):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(act[0, :, :, i], cmap="gray")
            ax.set_title(f"{name}[{i}]")
            ax.axis("off")

        plt.suptitle(f"Feature Maps - {name}", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"{ds_name}_featuremaps_{name}.png"), dpi=300)
        plt.show()
        plt.close()

# Use a sample from test set
sample_idx = 0
visualize_feature_maps(model, x_test[sample_idx])


# ============================================================
# STEP 10) Confusion matrix + evaluation
# ============================================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation=45, values_format="d")
plt.title(f"Confusion Matrix - {ds_name}")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, f"{ds_name}_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()


# ============================================================
# STEP 11) Save model + Convert to TFLite
# ============================================================
final_keras_path = os.path.join(SAVE_DIR, f"final_{ds_name}.keras")
savedmodel_path = os.path.join(SAVE_DIR, f"final_{ds_name}_savedmodel")
tflite_path = os.path.join(SAVE_DIR, f"final_{ds_name}.tflite")

# Save Keras format (recommended)
model.save(final_keras_path)

# Export SavedModel (useful for serving / deployment)
model.export(savedmodel_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("\nSaved artifacts:")
print("Keras:", final_keras_path)
print("SavedModel:", savedmodel_path)
print("TFLite:", tflite_path)
print("\nDone ✅")