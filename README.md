# MNIST Handwritten Digit Recognition

MSI 5102 Type 2 Project -- Classify handwritten digits (0-9) using multiple machine learning models on the MNIST dataset, with hyperparameter optimization, dimensionality reduction visualization, and model comparison.

## Project Structure

```
MNIST-MSI/
├── mnist_digit_recognition.ipynb          # Main notebook (source)
├── mnist_digit_recognition_executed.ipynb  # Notebook with outputs
├── pyproject.toml                         # Project metadata & dependencies (uv)
├── requirements.txt                       # Pip-compatible dependencies
├── .python-version                        # Python version (3.12)
├── Requirment/                            # Course requirement PDFs
│   ├── Type 2_Projects.pdf
│   └── Type2_Project_Guidelines.pdf
├── sample_images.png                      # Sample digit images (0-9)
├── class_distribution.png                 # Train/test class distribution
├── knn_k_vs_accuracy.png                  # KNN hyperparameter search
├── lr_c_vs_accuracy.png                   # Logistic Regression C search
├── nn_training_curves.png                 # MLP & CNN loss/accuracy curves
├── confusion_matrices.png                 # Confusion matrices for all models
├── model_comparison.png                   # Accuracy/time comparison chart
├── dimensionality_reduction.png           # PCA & t-SNE 2D scatter plots
└── pca_variance.png                       # PCA explained variance analysis
```

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- macOS with Apple Silicon (MPS) / Linux with NVIDIA GPU (CUDA) / CPU

## Setup & Run

### Using uv (recommended)

```bash
# Clone and enter the project
cd MNIST-MSI

# Install dependencies and create virtual environment
uv sync

# Launch Jupyter Notebook
uv run jupyter notebook mnist_digit_recognition.ipynb
```

### Using pip

```bash
pip install -r requirements.txt
jupyter notebook mnist_digit_recognition.ipynb
```

### Run without GUI (headless execution)

```bash
uv run jupyter nbconvert --to notebook --execute mnist_digit_recognition.ipynb \
    --output mnist_digit_recognition_executed.ipynb \
    --ExecutePreprocessor.timeout=600
```

## Notebook Structure & Code Logic

The notebook is organized into 6 sections:

### 1. Environment Setup & Data Loading

- Import libraries (numpy, pandas, sklearn, torch, matplotlib, seaborn)
- Auto-detect compute device: Apple MPS > CUDA GPU > CPU
- Load MNIST via `sklearn.datasets.fetch_openml('mnist_784')` (70,000 images, 28x28 pixels)
- Normalize pixel values to [0, 1], split into 60,000 train / 10,000 test

### 2. Data Visualization

- Display 5 sample images for each digit class (10x5 grid)
- Plot class distribution bar charts for train/test sets

### 3. Model Implementation

#### 3a. k-Nearest Neighbors (KNN)

- **Algorithm**: Classifies by majority vote of k nearest training samples
- **Pipeline**: `Pipeline(PCA + KNN)` -- PCA is fitted only within each CV fold, preventing data leakage
- **Optimization pipeline**:
  1. GridSearchCV over PCA components (30, 50, 100), k (1,3,5,7,9), distance metric (euclidean, manhattan), and voting weights (uniform, distance)
  2. Train final pipeline with best parameters on full training set
- **Output**: k vs. accuracy curve for all configurations

#### 3b. Logistic Regression

- **Algorithm**: Multinomial logistic regression with softmax for 10-class classification
- **Pipeline**: Independent `Pipeline(PCA + LR)` with its own PCA search space (separate from KNN), PCA fitted within each CV fold
- **Optimization pipeline**:
  1. GridSearchCV over PCA components (30, 50, 100, 150), regularization strength C (0.01, 0.1, 1.0, 10.0), and solver (lbfgs, saga)
  2. L2 penalty for regularization
  3. Train final pipeline with best parameters (max_iter=2000)
- **Output**: C vs. accuracy curve for different solvers

#### 3c. Neural Network (Bonus)

Two PyTorch architectures with shared training infrastructure:

**MLP** (Multi-Layer Perceptron):
```
784 -> Linear(512) -> BatchNorm -> ReLU -> Dropout(0.3)
    -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
    -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)
    -> Linear(10)
```

**CNN** (Convolutional Neural Network):
```
Input(1x28x28)
-> Conv2d(32, 3x3) -> BN -> ReLU -> Conv2d(32, 3x3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)
-> Conv2d(64, 3x3) -> BN -> ReLU -> Conv2d(64, 3x3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)
-> Flatten -> Linear(256) -> BN -> ReLU -> Dropout(0.5) -> Linear(10)
```

**Optimization techniques**:
| Technique | MLP | CNN | Purpose |
|-----------|-----|-----|---------|
| BatchNorm + Dropout | Yes | Yes | Regularization & faster convergence |
| Data augmentation | Gaussian noise (std=0.05) | Rotation +-10deg, Translation +-10% | Improve generalization |
| Adam + weight_decay(1e-4) | Yes | Yes | L2 regularization |
| ReduceLROnPlateau | patience=3, factor=0.5 | Same | Adaptive learning rate |
| Early stopping | patience=5 | Same | Prevent overfitting |

**Device compatibility**: Automatically uses `torch.device("mps")` on MacBook Apple Silicon, `torch.device("cuda")` on Kaggle/NVIDIA, or CPU as fallback.

### 4. Model Evaluation & Comparison

- Per-model: accuracy, precision/recall/F1 (classification report), confusion matrix heatmap
- Summary comparison table: accuracy, training time, prediction time
- Bar chart visualization of all metrics
- Discussion of strengths/weaknesses for each model

### 5. Dimensionality Reduction & Visualization

- **PCA**: Project 10,000 samples to 2D, analyze explained variance ratio and cumulative variance curve
- **t-SNE**: Apply PCA(50D) first for speed, then t-SNE(2D) with perplexity=30
- Side-by-side scatter plots colored by digit class
- Interpretation of clustering patterns and digit confusion

### 6. Optimization Summary

- Full table of all optimization techniques applied per model
- Final performance comparison

## Results

| Model | Test Accuracy | Training Time |
|-------|--------------|---------------|
| CNN (Neural Network) | **99.58%** | ~120s |
| MLP (Neural Network) | 98.62% | ~30s |
| KNN (Optimized) | 97.65% | ~2s |
| Logistic Regression | 92.25% | ~5s |

> Training times measured on MacBook with Apple Silicon (MPS). Times vary by hardware.

### Key Findings

- **CNN achieves the highest accuracy (99.58%)** by leveraging spatial structure of images through convolutional filters, combined with per-sample data augmentation (rotation/translation)
- **KNN with PCA + distance-weighted voting** reaches 97.65% with minimal complexity -- PCA(30D) reduces computation while preserving discriminative information
- **Logistic Regression** with independent PCA(150D) pipeline achieves 92.25%, higher than without PCA optimization, but still limited by its linear decision boundary
- **t-SNE** produces much clearer digit clusters than PCA in 2D, revealing which digits are visually similar (e.g., 4/9, 3/5)
- All CV pipelines use `sklearn.pipeline.Pipeline` to prevent PCA data leakage across folds

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, pandas | Data manipulation |
| matplotlib, seaborn | Visualization |
| scikit-learn | KNN, Logistic Regression, PCA, t-SNE, metrics |
| torch, torchvision | Neural Networks (MLP, CNN), data augmentation |
| jupyter, ipykernel | Notebook execution |
