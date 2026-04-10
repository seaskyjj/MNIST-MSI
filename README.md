# MNIST Handwritten Digit Recognition

MSI 5102 Type 2 Project: classify handwritten digits (`0-9`) on MNIST using classical machine-learning baselines and neural networks, with full hyperparameter search, dimensionality-reduction analysis, and experiment reporting.

## Current Version

The current primary notebook is [mnist_full_search_and_training.ipynb](/Users/jiejia/Programs/MNIST-MSI/mnist_full_search_and_training.ipynb).

This notebook includes:

- full-training-set `GridSearchCV` for KNN and Logistic Regression
- a new KNN `k vs CV accuracy` visualization
- saved search artifacts in [`output/`](/Users/jiejia/Programs/MNIST-MSI/output)
- configurable MLP and CNN training logic
- best-checkpoint restoration for neural models
- PCA and t-SNE visualization

The older notebooks are retained as legacy/reference notebooks:

- [mnist_digit_recognition.ipynb](/Users/jiejia/Programs/MNIST-MSI/mnist_digit_recognition.ipynb)
- [mnist_digit_recognition_executed.ipynb](/Users/jiejia/Programs/MNIST-MSI/mnist_digit_recognition_executed.ipynb)

## Project Structure

```text
MNIST-MSI/
├── mnist_full_search_and_training.ipynb   # Current main notebook
├── mnist_digit_recognition.ipynb          # Legacy source notebook
├── mnist_digit_recognition_executed.ipynb # Legacy executed notebook
├── output/                                # Figures, CSVs, search summaries, serialized estimators
├── docs/report.md                         # Current experiment report
├── docs/notes.md                          # Project notes
├── docs/review.md                         # Review findings
├── pyproject.toml                         # Project metadata and uv dependencies
├── requirements.txt                       # Minimal pip dependencies
└── Requirment/                            # Course requirement PDFs
```

## Environment

- Python `>= 3.12`
- `uv` recommended
- macOS with Apple Silicon (`mps`), Linux with CUDA, or CPU

## Setup and Run

### Using uv

```bash
cd MNIST-MSI
uv sync
uv run jupyter notebook mnist_full_search_and_training.ipynb
```

### Using pip

`requirements.txt` currently contains only the core ML libraries. If you use pip, install Jupyter separately:

```bash
pip install -r requirements.txt jupyter ipykernel
jupyter notebook mnist_full_search_and_training.ipynb
```

### Headless Execution

```bash
uv run jupyter nbconvert --to notebook --execute mnist_full_search_and_training.ipynb \
    --output mnist_full_search_and_training.executed.ipynb \
    --ExecutePreprocessor.timeout=7200
```

## Notebook Workflow

The current notebook is organized around the following stages:

1. Environment setup and reproducibility configuration
2. MNIST loading, normalization, and train/test split
3. Sample-image and class-distribution visualization
4. Full `GridSearchCV` for KNN
5. Full `GridSearchCV` for Logistic Regression
6. Final training and evaluation of the selected classical models
7. Validation split and data pipeline setup for neural networks
8. MLP and CNN training with checkpoint restoration
9. Final comparison table and confusion matrices
10. PCA and t-SNE visualization
11. PCA variance analysis

## Model Details

### KNN

Implementation:

- `Pipeline(PCA -> KNeighborsClassifier)`
- PCA is fitted inside each CV fold, avoiding leakage

Search space:

- `pca__n_components = [0.90, 0.95, 0.99]`
- `knn__n_neighbors = [1, 3, 5, 7, 9]`
- `knn__weights = ['uniform', 'distance']`
- `knn__metric = ['euclidean', 'manhattan']`

Search configuration:

- full training set (`60,000` samples)
- `5`-fold cross-validation
- `60` parameter combinations
- `300` total fits

Best KNN configuration:

- `metric='euclidean'`
- `n_neighbors=3`
- `weights='distance'`
- `pca__n_components=0.90`

### Logistic Regression

Implementation:

- `Pipeline(PCA -> LogisticRegression)`
- independent PCA search space, not reused from KNN

Search space:

- `pca__n_components = [0.90, 0.95, 0.99]`
- `lr__C = [0.01, 0.1, 1.0, 10.0]`
- `lr__solver = ['lbfgs', 'saga']`
- `lr__penalty = ['l2']`
- `max_iter = 2000`

Search configuration:

- full training set (`60,000` samples)
- `5`-fold cross-validation
- `24` parameter combinations
- `120` total fits

Best Logistic Regression configuration:

- `C=0.1`
- `solver='lbfgs'`
- `penalty='l2'`
- `pca__n_components=0.99`

### Neural Networks

Both neural models use a shared training function with:

- checkpoint restoration based on the best validation accuracy
- `ReduceLROnPlateau(patience=3, factor=0.5)` on validation loss
- configurable early-stopping logic

The latest saved notebook outputs correspond to the override cell immediately before neural training:

- optimizer: `AdamW`
- learning rate: `0.0005`
- weight decay: `1e-4`
- epochs: `50`
- early stopping: `False`
- restore best weights: `True`

#### MLP

Architecture:

```text
784
-> Linear(512) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)
-> Linear(10)
```

Training augmentation:

- inline Gaussian noise with `std=0.05`

#### CNN

Architecture:

```text
Input(1x28x28)
-> Conv2d(32, 3x3) -> BN -> ReLU
-> Conv2d(32, 3x3) -> BN -> ReLU
-> MaxPool(2) -> Dropout(0.25)
-> Conv2d(64, 3x3) -> BN -> ReLU
-> Conv2d(64, 3x3) -> BN -> ReLU
-> MaxPool(2) -> Dropout(0.25)
-> Flatten
-> Linear(256) -> BN -> ReLU -> Dropout(0.5)
-> Linear(10)
```

Training augmentation:

- `RandomRotation(±10°)`
- `RandomAffine(translate=10%)`

## Current Results

### Final Test-Set Results

| Model | Test Accuracy | Final Train Time | Predict Time |
| --- | ---: | ---: | ---: |
| CNN (Neural Network) | **99.65%** | `347.29s` | `0.0663s` |
| MLP (Neural Network) | `98.72%` | `32.19s` | `0.0241s` |
| KNN (Optimized) | `97.74%` | `0.2167s` | `0.2526s` |
| Logistic Regression (Optimized) | `92.27%` | `0.8647s` | `0.0079s` |

Important note:

- for KNN and Logistic Regression, the table reports final fit time after model selection
- `GridSearchCV` search time is reported separately below and is not included in the final-train-time column

### Classical Model Search Summary

| Model | Best CV Accuracy | Search Time | Best Params |
| --- | ---: | ---: | --- |
| KNN | `0.9758` | `510.55s` | `euclidean`, `k=3`, `distance`, `pca__n_components=0.90` |
| Logistic Regression | `0.9216` | `626.57s` | `C=0.1`, `lbfgs`, `l2`, `pca__n_components=0.99` |

## Key Findings

- CNN is the strongest model in the current version, reaching `99.65%` test accuracy.
- MLP reaches `98.72%` and remains a strong non-convolutional baseline.
- KNN is a competitive classical baseline, especially with Euclidean distance, small `k`, and `90%` PCA retention.
- Logistic Regression is clearly weaker than the other three models, which is consistent with the limited expressiveness of linear decision boundaries on MNIST.
- Full `Pipeline`-based CV prevents PCA leakage in both KNN and Logistic Regression.

## PCA and t-SNE

The dimensionality-reduction section uses a `10,000`-sample subset of the training data.

PCA:

- first two principal components explain about `16.70%` of total variance
- the 2D PCA plot reveals broad global structure but substantial class overlap
- this supports the conclusion that MNIST is not well separated in a very low-dimensional linear subspace

t-SNE:

- t-SNE produces much clearer local class clusters than PCA
- this suggests that MNIST has strong local neighborhood structure that nonlinear methods can exploit
- this is consistent with the model ranking: `Logistic Regression < KNN < MLP < CNN`

PCA variance note:

- the first `50` principal components explain about `82.61%` cumulative variance
- `95%` variance is **not** reached within the first `50` components
- the notebook logic was corrected so it no longer incorrectly reports `1 component for 95%`

## Saved Outputs

The current notebook writes generated artifacts to [`output/`](/Users/jiejia/Programs/MNIST-MSI/output), including:

- figures:
  - `knn_gridsearch_all_results_v2.png`
  - `knn_gridsearch_lineplot_v2.png`
  - `knn_k_vs_cv_accuracy_v2.png`
  - `lr_gridsearch_all_results_v2.png`
  - `lr_gridsearch_lineplot_v2.png`
  - `nn_training_curves_v2.png`
  - `model_comparison_v2.png`
  - `confusion_matrices_v2.png`
  - `dimensionality_reduction.png`
  - `pca_variance.png`
- search artifacts:
  - `knn_cv_results_raw_v2.csv`
  - `knn_cv_results_full_v2.csv`
  - `knn_cv_results_ranked_v2.csv`
  - `knn_search_summary_v2.json`
  - `knn_best_estimator_v2.joblib`
  - `lr_cv_results_raw_v2.csv`
  - `lr_cv_results_full_v2.csv`
  - `lr_cv_results_ranked_v2.csv`
  - `lr_search_summary_v2.json`
  - `lr_best_estimator_v2.joblib`

## Documentation

- [report.md](/Users/jiejia/Programs/MNIST-MSI/docs/report.md): experiment report
- [notes.md](/Users/jiejia/Programs/MNIST-MSI/docs/notes.md): implementation notes
- [review.md](/Users/jiejia/Programs/MNIST-MSI/docs/review.md): review findings and fixes

## Dependencies

Core packages used in the project:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `torchvision`
- `jupyter`
- `ipykernel`
