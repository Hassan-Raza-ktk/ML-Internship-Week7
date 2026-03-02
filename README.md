# 🧠 ML Internship – Week 7  
**Neuro App Internship – Deep Learning Specialization**

This repository contains Week 7 implementations focused on Neural Networks, CNNs, and Transfer Learning using TensorFlow & Keras.

---

# 📌 Overview

This week covers:

| Task | Topic |
|------|-------|
| 7.1 | Multi-Layer Perceptron (MLP) from Scratch (XOR) |
| 7.2 | Neural Network using Keras |
| 7.3 | CNN Image Classification |
| 7.4 | Transfer Learning with MobileNetV2 + Fine-Tuning |

---

# 🧪 Task 7.1 — MLP From Scratch (XOR)

Implemented a full neural network pipeline manually:

- Random weight initialization  
- Sigmoid activation  
- Forward propagation  
- Cross-entropy loss  
- Backpropagation  
- Gradient descent  
- Decision boundary visualization  

### 📊 Training Loss Curve

![Loss Curve](Task_7.1_Multi_Layer_Perceptron/visuals/loss_curve.png)

### 📈 Decision Boundary

![Decision Boundary](Task_7.1_Multi_Layer_Perceptron/visuals/decision_boundary.png)

---

# 🤖 Task 7.2 — Neural Network using Keras

Rebuilt XOR using Keras:

- Dense layers
- Binary cross-entropy
- Adam optimizer
- Early stopping
- Model saving (.keras / .h5)

Model summary shows:

- 65 trainable parameters
- Hidden layers with nonlinear activations

---

# 🧩 Task 7.3 — CNN Image Classification

Built CNN from scratch:

- Conv2D
- MaxPooling
- Flatten
- Dense
- Softmax output

### 📊 Training Accuracy vs Validation Accuracy

![Accuracy Curve](Task7.3_CNN_Image_Classification/visuals/accuracy_curve.png)

### 📉 Training Loss vs Validation Loss

![Loss Curve](Task7.3_CNN_Image_Classification/visuals/loss_curve.png)

---

# 🚀 Task 7.4 — Transfer Learning (MobileNetV2)

Used pre-trained MobileNetV2 with:

- `include_top=False`
- Frozen base layers
- Custom classification head
- Fine-tuning (partial unfreeze)
- Lower learning rate retraining
- Scratch model comparison

---

## 🔬 Performance Comparison

| Model | Validation Accuracy |
|-------|---------------------|
| Transfer Learning (Frozen) | ~90% |
| Fine-Tuned Model | ~92% |
| Trained From Scratch | ~55% |

---

## 📊 Accuracy Comparison Chart

![Model Comparison](Task_7.4_Transfer_Learning/visuals/model_comparison.png)

---

# 💾 Model Saving Formats

Models exported in multiple formats:

- `.keras`
- `.h5`
- `.tflite`
- SavedModel directory

---

# ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
