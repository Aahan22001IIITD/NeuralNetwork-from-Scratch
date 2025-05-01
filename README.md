# NeuralNetwork-from-Scratch
# 🧠 Neural Network from Scratch for MNIST Classification

This project implements a flexible, fully customizable feedforward neural network **from scratch** (no deep learning libraries) to classify handwritten digits from the MNIST dataset. The model supports configurable architecture, activation functions, weight initialization strategies, and optimization settings — offering an educational deep dive into how neural networks work at the core.

---

## 📌 Objectives

- Build a neural network class capable of handling arbitrary architectures.
- Implement multiple activation and weight initialization functions.
- Train the network on MNIST and analyze performance across configurations.
- Save trained models and provide reproducibility via `.pkl` files.

---

## 🔧 Features

### 🏗️ `NeuralNetwork` Class
```python
NeuralNetwork(
    N,              # Number of layers
    layers,         # List of neurons in each layer
    lr,             # Learning rate
    activation,     # Activation function (same for all except last)
    weight_init,    # Weight initialization method
    epochs,         # Number of training epochs
    batch_size      # Mini-batch size
)```
✅ Core Methods
fit(X, Y): Train the network.

predict(X): Return predicted class labels.

predict_proba(X): Return class-wise probabilities.

score(X, Y): Return accuracy score on input data.

⚙️ Supported Activation Functions
Sigmoid

Tanh

ReLU

Leaky ReLU

Softmax (used only in output layer)

Each includes forward and backward (gradient) implementations.

🧪 Weight Initialization Options
Zero Initialization

Random Uniform Initialization

Normal Initialization (𝒩(0, 1) with proper scaling)

🧠 Training Configuration
Parameter	Value
Hidden Layers	4
Layer Sizes	[256, 128, 64, 32]
Learning Rate	2e-5
Epochs	100 (with early stopping)
Batch Size	128
Dataset	MNIST (80/10/10 split)

📊 Evaluation
12 total model configurations (4 activations × 3 initializations)

Plotted training and validation loss per epoch

Compared convergence behavior and final accuracy

Best-performing combinations identified and analyzed

💾 Deliverables
NeuralNetwork.py: Main implementation

activations.py: All activation functions and gradients

initializers.py: Weight initialization strategies

train.py: Script to train and evaluate the models

plots/: Training & validation loss plots

models/: 12 saved models in .pkl format

report.pdf: Summary of findings and analysis

📝 Results Snapshot
Activation	Init Method	Accuracy (Test)	Comments
ReLU	Normal	✅ High	Best performance
Tanh	Zero	❌ Low	Vanishing gradients
Leaky ReLU	Random	✅ Moderate	Stable training
