# MNIST-MSI Experiment Report

## 1. Objective

This report summarizes the current experimental workflow and results in `mnist_full_search_and_training.ipynb`, including:

- end-to-end model training and evaluation on MNIST
- full-training-set `GridSearchCV` for KNN and Logistic Regression
- MLP and CNN training under the latest saved notebook configuration
- PCA and t-SNE visualization results
- a partial ablation study on fixed neural-network architectures

## 2. Dataset and Workflow

The notebook uses the MNIST handwritten-digit dataset from `fetch_openml('mnist_784')`.

The overall workflow is:

1. Load MNIST and normalize pixel values to `[0, 1]`.
2. Split the dataset into `60,000` training samples and `10,000` test samples with stratification.
3. Visualize sample digits and class distribution.
4. Run full `GridSearchCV` on the entire training set for:
   - KNN with a PCA + KNN pipeline
   - Logistic Regression with a PCA + Logistic Regression pipeline
5. Split the training set again into:
   - `54,000` neural-network training samples
   - `6,000` validation samples
6. Train:
   - an MLP on flattened images
   - a CNN with image-space augmentation
7. Restore the best validation checkpoint for each neural model.
8. Compare all four models on the held-out test set.
9. Run PCA and t-SNE for qualitative visualization.

## 3. Saved Notebook Configuration Used for the Latest Neural-Network Outputs

The current notebook contains an override cell immediately before neural-network training. The saved MLP/CNN outputs in the notebook correspond to the following configuration:


| Setting              | Value                                      |
| -------------------- | ------------------------------------------ |
| Optimizer            | `AdamW`                                    |
| Learning rate        | `0.0005`                                   |
| Weight decay         | `1e-4`                                     |
| Epochs               | `50`                                       |
| Early stopping       | `False`                                    |
| Restore best weights | `True`                                     |
| Checkpoint rule      | Save on any strict improvement in`val_acc` |
| Early-stop threshold | `min_delta = 0.000333 (2/6000)`            |

This means the latest saved neural-network results are based on full training for 50 epochs, followed by restoration of the best validation checkpoint.

## 4. Traditional Models

### 4.1 KNN

The KNN model is implemented as a `Pipeline(PCA -> KNeighborsClassifier)` and tuned on the full `60,000`-sample training set using 5-fold cross-validation.

Search space:

- `pca__n_components = [0.90, 0.95, 0.99]`
- `n_neighbors = [1, 3, 5, 7, 9]`
- `weights = ['uniform', 'distance']`
- `metric = ['euclidean', 'manhattan']`

Search scale:

- `60` parameter combinations
- `300` total fits (`60 combinations x 5 folds`)

Best configuration:


| Item                | Value                                                                           |
| ------------------- | ------------------------------------------------------------------------------- |
| Best params         | `metric='euclidean', n_neighbors=3, weights='distance', pca__n_components=0.90` |
| Best CV accuracy    | `0.9758`                                                                        |
| Search time         | `510.55 s`                                                                      |
| Internal refit time | `0.22 s`                                                                        |
| Final test accuracy | `97.74%`                                                                        |
| Final train time    | `0.2167 s`                                                                      |
| Prediction time     | `0.2526 s`                                                                      |

Key observations from the full search table:

- Euclidean distance consistently outperformed Manhattan distance.
- Retaining `90%` PCA variance was better than retaining `95%` or `99%`.
- Distance-weighted voting slightly outperformed uniform voting in the best-performing region.
- The top-ranked KNN configurations were tightly clustered, suggesting that KNN is relatively stable once the metric and PCA retention are chosen well.

Saved search artifacts:

- `output/knn_cv_results_raw_v2.csv`
- `output/knn_cv_results_full_v2.csv`
- `output/knn_cv_results_ranked_v2.csv`
- `output/knn_search_summary_v2.json`
- `output/knn_best_estimator_v2.joblib`

### 4.2 Logistic Regression

The Logistic Regression model is implemented as a `Pipeline(PCA -> LogisticRegression)` and also tuned on the full training set with 5-fold cross-validation.

Search space:

- `pca__n_components = [0.90, 0.95, 0.99]`
- `C = [0.01, 0.1, 1.0, 10.0]`
- `solver = ['lbfgs', 'saga']`
- `penalty = ['l2']`

Search scale:

- `24` parameter combinations
- `120` total fits (`24 combinations x 5 folds`)

Best configuration:


| Item                | Value                                                         |
| ------------------- | ------------------------------------------------------------- |
| Best params         | `C=0.1, solver='lbfgs', penalty='l2', pca__n_components=0.99` |
| Best CV accuracy    | `0.9216`                                                      |
| Search time         | `626.57 s`                                                    |
| Internal refit time | `0.87 s`                                                      |
| Final test accuracy | `92.27%`                                                      |
| Final train time    | `0.8647 s`                                                    |
| Prediction time     | `0.0079 s`                                                    |

Key observations from the full search table:

- `pca__n_components = 0.99` dominated the highest-ranked configurations.
- `lbfgs` and `saga` produced very similar validation accuracy near the top of the table.
- `lbfgs` was substantially faster than `saga` for comparable accuracy, making it the better practical choice here.
- Logistic Regression remained clearly below KNN and both neural models, indicating that a linear decision boundary is limiting on this task even after PCA.

Saved search artifacts:

- `output/lr_cv_results_raw_v2.csv`
- `output/lr_cv_results_full_v2.csv`
- `output/lr_cv_results_ranked_v2.csv`
- `output/lr_search_summary_v2.json`
- `output/lr_best_estimator_v2.joblib`

### 4.3 How to Interpret the Full GridSearchCV Results

Because the notebook uses `scoring='accuracy'`, the `mean_test_score` in the search tables is the mean cross-validation accuracy across the 5 folds.

Practical interpretation:

- `rank_test_score = 1` means that this parameter combination ranked first among all combinations.
- The corresponding `mean_test_score` is the best cross-validation accuracy.
- `std_test_score` indicates how much that score varies across folds.
- `grid.best_score_` is the same value as the top-ranked row's `mean_test_score`.

Important distinction:

- `mean_test_score` is cross-validation accuracy on the training set.
- It is not the final held-out test accuracy.
- Final test accuracy is reported only after refitting the selected model on the full training set and evaluating on `X_test`.

## 5. Neural Networks

### 5.1 MLP

Architecture:

- `784 -> 512 -> 256 -> 128 -> 10`
- `BatchNorm + ReLU + Dropout` after each hidden layer
- Dropout rates: `0.3`, `0.3`, `0.2`

Training details:

- Input format: flattened `28 x 28` images
- Optimizer: `AdamW`
- Learning rate: `0.0005`
- Weight decay: `1e-4`
- Scheduler: `ReduceLROnPlateau(patience=3, factor=0.5)` on validation loss
- Data augmentation: Gaussian input noise with standard deviation `0.05`
- Epochs: `50`
- Best-checkpoint restoration: enabled
- Early stopping: disabled in the current saved run

Latest saved result:


| Item                     | Value      |
| ------------------------ | ---------- |
| Best checkpoint epoch    | `45`       |
| Best validation accuracy | `0.9875`   |
| Test accuracy            | `98.72%`   |
| Training time            | `32.19 s`  |
| Prediction time          | `0.0241 s` |

Interpretation:

- The MLP improved materially when training was extended and the best validation checkpoint was restored at the end.
- Compared with the earlier 40-epoch runs, the current 50-epoch notebook result indicates that the MLP still benefited from additional optimization steps under the current setup.

### 5.2 CNN

Architecture:

- Two convolutional blocks with `32` and `64` channels
- Each block uses `Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout`
- Classifier head: `Linear(3136 -> 256) -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear(256 -> 10)`

Training details:

- Input format: image-shaped tensors reshaped from flattened vectors
- Optimizer: `AdamW`
- Learning rate: `0.0005`
- Weight decay: `1e-4`
- Scheduler: `ReduceLROnPlateau(patience=3, factor=0.5)` on validation loss
- Data augmentation:
  - `RandomRotation(±10°)`
  - `RandomAffine(translate=10%)`
- Epochs: `50`
- Best-checkpoint restoration: enabled
- Early stopping: disabled in the current saved run

Latest saved result:


| Item                     | Value      |
| ------------------------ | ---------- |
| Best checkpoint epoch    | `44`       |
| Best validation accuracy | `0.9978`   |
| Test accuracy            | `99.65%`   |
| Training time            | `347.29 s` |
| Prediction time          | `0.0663 s` |

Interpretation:

- The CNN remained the strongest model in the project.
- Validation accuracy reached a stable high plateau in the late stages of training, and restoring the best checkpoint delivered the strongest final test result among all models.
- The CNN significantly outperformed both traditional models and the MLP on the held-out test set.

## 6. Final Test-Set Comparison


| Model                           | Test Accuracy | Train Time (s) | Predict Time (s) |
| ------------------------------- | ------------: | -------------: | ---------------: |
| CNN (Neural Network)            |      `99.65%` |     `347.2937` |         `0.0663` |
| MLP (Neural Network)            |      `98.72%` |      `32.1945` |         `0.0241` |
| KNN (Optimized)                 |      `97.74%` |       `0.2167` |         `0.2526` |
| Logistic Regression (Optimized) |      `92.27%` |       `0.8647` |         `0.0079` |

Overall conclusion:

- The CNN achieved the best overall accuracy.
- The MLP provided a strong performance/complexity trade-off.
- KNN was competitive and strong for a classical method, but slower at inference than the learned linear model.
- Logistic Regression served as a useful baseline but lagged behind due to its limited linear expressiveness.

## 7. Dimensionality Reduction and Visualization

### 7.1 PCA and t-SNE Setup

The visualization analysis uses a random `10,000`-sample subset drawn from the training set. Two complementary projection methods are applied:

- PCA is applied directly to obtain a 2-dimensional linear projection.
- t-SNE is applied after an intermediate 50-dimensional PCA reduction for computational efficiency.

The notebook reports the following PCA statistics for the 2-dimensional projection:

- first principal component explained variance ratio: `0.0954`
- second principal component explained variance ratio: `0.0716`
- cumulative variance explained by the first two components: `0.1670`

These values indicate that the first two principal components retain only a modest fraction of the total variation in the original 784-dimensional image space. Therefore, the 2D PCA plot should be interpreted as a coarse structural summary rather than a near-lossless representation of the dataset.

### 7.2 Visual Interpretation

#### PCA

The PCA projection reveals broad global organization but substantial class overlap. Several digit classes exhibit directional tendencies in the 2D plane, yet the class clouds remain heavily entangled in the center of the projection. This is especially visible for digits with similar stroke patterns, such as `2`, `3`, `5`, `8`, and `9`.

From an analytical perspective, the PCA plot supports the following interpretation:

- MNIST contains meaningful low-dimensional global structure.
- That global structure is not sufficient, in only two linear components, to separate classes cleanly.
- A purely linear low-dimensional view does not capture the full discriminative geometry of the dataset.

This observation is consistent with the model results. Logistic Regression, which relies on linear decision boundaries after PCA, performs substantially worse than KNN, MLP, and CNN.

#### t-SNE

The t-SNE projection produces a much clearer organization of the data. Most digit classes form compact and visually coherent clusters, with only limited overlap between morphologically similar classes. Compared with PCA, t-SNE makes local neighborhood structure much more explicit.

This suggests that:

- in the original high-dimensional feature space, samples from the same class tend to form locally coherent neighborhoods
- class separation is more apparent in a nonlinear manifold view than in a linear 2D projection
- methods that exploit local or nonlinear structure are better aligned with the geometry of MNIST

This interpretation is again consistent with the empirical model ranking:

- Logistic Regression is weakest
- KNN is substantially stronger because it leverages local neighborhood information
- MLP and CNN outperform both classical baselines because they learn nonlinear decision boundaries
- CNN performs best because it also exploits the spatial structure of the image

#### Caution in Interpretation

The PCA and t-SNE plots should not be treated as equivalent forms of evidence:

- PCA preserves variance in a linear sense and is more informative about global structure.
- t-SNE preserves local neighborhood relationships and is more informative about cluster formation.
- Distances and cluster spacing in t-SNE should not be interpreted as exact global geometric relationships in the original space.

Accordingly, the most defensible conclusion is not that PCA is "bad" and t-SNE is "good," but rather that the two projections answer different questions. PCA shows that the dataset is not easily separable in a very low-dimensional linear subspace, while t-SNE shows that the dataset contains strong local class structure that nonlinear methods can exploit.


## 8. Partial Ablation on Fixed Architecture

This section summarizes 5 runs  which keep the network architectures fixed and use the same "restore best checkpoint" strategy. These runs are useful as a partial ablation on optimizer and weight decay, but they are still single-run comparisons rather than a full multi-seed ablation study.

Run 1 is intentionally excluded from this section because it changed more than one factor at once and therefore does not provide a clean controlled comparison.

### 8.1 Partial-Ablation Table


| Run | Checkpoint Rule | Optimizer | Weight Decay | MLP Best Epoch / Val Acc | MLP Test Acc | CNN Best Epoch / Val Acc | CNN Test Acc |
| --- | --------------- | --------- | -----------: | ------------------------ | -----------: | ------------------------ | -----------: |
| 1   | Best checkpoint | Adam      |       `5e-5` | `40 / 0.9870`            |     `98.65%` | `28 / 0.9970`            |     `99.57%` |
| 2   | Best checkpoint | AdamW     |       `5e-5` | `39 / 0.9875`            |     `98.49%` | `38 / 0.9973`            |     `99.57%` |
| 3   | Best checkpoint | AdamW     |          `0` | `22 / 0.9878`            |     `98.46%` | `35 / 0.9978`            |     `99.59%` |
| 4   | Best checkpoint | AdamW     |       `1e-5` | `30 / 0.9882`            |     `98.54%` | `35 / 0.9977`            |     `99.58%` |
| 5   | Best checkpoint | AdamW     |       `1e-4` | `23 / 0.9877`            |     `98.60%` | `30 / 0.9972`            |     `99.60%` |

### 8.2 Findings from the Partial Ablation

MLP:

- At fixed `weight_decay = 5e-5`, Adam outperformed AdamW in this single-run comparison:
  - Adam: `98.65%`
  - AdamW: `98.49%`
- Within the AdamW weight-decay sweep, `1e-4` was the best test-setting among runs `2 / 3 / 4 / 5`:
  - `0`: `98.46%`
  - `1e-5`: `98.54%`
  - `5e-5`: `98.49%`
  - `1e-4`: `98.60%`
- The latest notebook run improved further to `98.72%`, which suggests that training budget and checkpoint selection matter in addition to optimizer choice.

CNN:

- The CNN was much less sensitive to optimizer and weight decay than the MLP.
- Across runs `1 / 2 / 3 / 4 / 5`, CNN test accuracy ranged only from `99.57%` to `99.60%`.
- At fixed `weight_decay = 5e-5`, Adam and AdamW produced the same CNN test accuracy (`99.57%`).
- Within the AdamW sweep, `1e-4` was marginally best (`99.60%`), but the absolute spread was tiny.

Interpretation:

- For the CNN, optimizer and weight decay produced only marginal differences under the tested range.
- For the MLP, regularization and optimizer choices had a larger effect, though the differences are still modest relative to the total accuracy.
- These results are informative, but they should be treated as exploratory rather than definitive because they are based on single runs rather than repeated runs across multiple seeds.

## 9. Main Conclusions

1. The CNN is the strongest model in the project and reaches `99.65%` test accuracy in the latest saved notebook run.
2. The MLP remains competitive at `98.72%`, especially after switching to best-checkpoint restoration and increasing the training budget.
3. KNN is a strong classical baseline with `97.74%` test accuracy; the best region is Euclidean distance, small `k`, and `90%` PCA retention.
4. Logistic Regression is much weaker than the other three approaches, peaking at `92.27%` test accuracy under the current search space.
5. The PCA view shows broad global structure but strong overlap, while t-SNE reveals much clearer local class clusters.
6. The partial ablation suggests that:
   - MLP is more sensitive than CNN to optimizer and weight decay.
   - CNN is comparatively robust across the tested regularization settings.
   - `AdamW + weight_decay = 1e-4` is a reasonable working choice for the current notebook setup.
