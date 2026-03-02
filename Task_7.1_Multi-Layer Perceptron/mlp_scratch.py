# mlp_scratch.py
# ============================================================
# Week 7 - Task 7.1: Multi-Layer Perceptron (MLP) from Scratch
# ============================================================
# Goal:
# - Implement a tiny neural network using ONLY NumPy
# - Train it on XOR (non-linear) and visualize learning
#
# Output:
# - Loss curve (loss vs epochs)
# - Decision boundary plot
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Keep runs consistent (useful for debugging + reporting)
np.random.seed(42)


# ============================================================
# STEP 1) Create XOR dataset (or use Iris)
# ============================================================
# XOR is the classic "you NEED a hidden layer" demo.
# Points:
#   (0,0) -> 0
#   (0,1) -> 1
#   (1,0) -> 1
#   (1,1) -> 0
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # shape: (4, 1)


# ============================================================
# STEP 2) Initialize weights randomly
# ============================================================
# Network shape:
#   input (2 features) -> hidden (H neurons) -> output (1 neuron)
#
# Why random init?
# - If all weights start same, all neurons learn same thing (bad).
# - Randomness breaks symmetry so different neurons learn different patterns.
input_size = 2
hidden_size = 6
output_size = 1

# Small random weights. Biases start at 0 (safe & common).
W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))


# ============================================================
# STEP 3) Implement sigmoid activation function
# ============================================================
# Sigmoid turns any number into a probability-like value (0..1).
# It's smooth and differentiable -> good for backprop.
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  # prevents exp overflow
    return 1.0 / (1.0 + np.exp(-z))


# We need derivative for backprop:
# If a = sigmoid(z) then da/dz = a*(1-a)
def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


# ============================================================
# STEP 4) Implement forward propagation
# ============================================================
# Forward pass math:
#   z1 = XW1 + b1
#   a1 = sigmoid(z1)
#   z2 = a1W2 + b2
#   a2 = sigmoid(z2)  -> final prediction
#
# We also store intermediate values (cache) for backprop.
def forward_pass(X_batch: np.ndarray):
    z1 = X_batch @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)  # prediction (0..1)

    cache = {
        "X": X_batch,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }
    return a2, cache


# ============================================================
# STEP 5) Implement cost function (cross-entropy)
# ============================================================
# Binary Cross-Entropy (BCE):
#   L = -[ y*log(p) + (1-y)*log(1-p) ]
#
# Why BCE?
# - It's the standard loss for binary classification.
# - Pairs well with sigmoid output.
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(loss))


# ============================================================
# STEP 6) Implement backpropagation
# ============================================================
# Backprop = chain rule in action.
#
# Key simplification:
# For sigmoid output + BCE loss:
#   dL/dz2 = (a2 - y)
#
# Then:
#   dW2 = a1^T * dz2
#   db2 = sum(dz2)
#
# Back to hidden:
#   da1 = dz2 * W2^T
#   dz1 = da1 * sigmoid'(z1)  (using a1*(1-a1))
#   dW1 = X^T * dz1
#   db1 = sum(dz1)
def backward_pass(y_true: np.ndarray, cache: dict):
    X_batch = cache["X"]
    a1 = cache["a1"]
    a2 = cache["a2"]

    m = X_batch.shape[0]  # number of samples

    # Output layer gradient
    dz2 = (a2 - y_true) / m               # shape: (m,1)
    dW2 = a1.T @ dz2                      # shape: (hidden,1)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer gradient
    da1 = dz2 @ W2.T                      # shape: (m,hidden)
    dz1 = da1 * sigmoid_derivative(a1)    # shape: (m,hidden)
    dW1 = X_batch.T @ dz1                 # shape: (2,hidden)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# ============================================================
# STEP 7) Implement weight update (gradient descent)
# ============================================================
# Gradient Descent update rule:
#   W = W - lr * dW
#   b = b - lr * db
#
# lr (learning rate) controls step size.
def apply_gradients(grads: dict, lr: float):
    global W1, b1, W2, b2
    W1 -= lr * grads["dW1"]
    b1 -= lr * grads["db1"]
    W2 -= lr * grads["dW2"]
    b2 -= lr * grads["db2"]


# ============================================================
# STEP 8) Create training loop for 1000 epochs
# ============================================================
epochs = 3000
lr = 0.3
loss_history = []

for epoch in range(1, epochs + 1):
    # forward pass
    y_pred, cache = forward_pass(X)

    # compute loss
    loss = binary_cross_entropy(y, y_pred)
    loss_history.append(loss)

    # backward pass
    grads = backward_pass(y, cache)

    # weight update
    apply_gradients(grads, lr)

    # tiny logs for progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


# Quick check after training
final_pred, _ = forward_pass(X)
print("\nFinal predictions (rounded):")
print(np.round(final_pred, 3))
print("Final classes:")
print((final_pred >= 0.5).astype(int).ravel())

pred_class = (final_pred >= 0.5).astype(int)
acc = (pred_class == y).mean()
print("Accuracy:", acc)

# ============================================================
# STEP 9) Plot loss vs epochs
# ============================================================

import os
        
# Make sure visuals folder exists
save_dir = os.path.join("Task_7.1_Multi-Layer Perceptron", "visuals")

os.makedirs(save_dir, exist_ok=True)
plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Epochs (XOR) - MLP from Scratch")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.tight_layout()

# Save high-quality image
save_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(save_path, dpi=300)

plt.show()




# ============================================================
# STEP 10) Visualize decision boundaries
# ============================================================
def plot_decision_boundary():
    # Grid area (a bit bigger than [0,1] so boundary looks nicer)
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    step = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs, _ = forward_pass(grid)
    probs = probs.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, probs, levels=50, alpha=0.85)
    plt.colorbar(label="P(class=1)")

    # XOR points on top
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors="k", s=120)

    for i, (x1, x2) in enumerate(X):
        plt.text(x1 + 0.03, x2 + 0.03, str(int(y[i, 0])), fontsize=12, weight="bold")

        import os

        # Make sure visuals folder exists
        save_dir = os.path.join("Task_7.1_Multi-Layer Perceptron", "visuals")
        os.makedirs(save_dir, exist_ok=True)

        plt.title("Decision Boundary (XOR)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()

        # Save high-quality image
        save_path = os.path.join(save_dir, "decision_boundary.png")
        plt.savefig(save_path, dpi=300)

        plt.show()


plot_decision_boundary()


# ============================================================
# STEP 11) Add detailed comments explaining math
# ============================================================
# Already done across the file:
# - forward pass equations (z1,a1,z2,a2)
# - BCE loss formula
# - backprop chain rule notes + key simplification (a2 - y)
# - gradient descent update rule
# ============================================================