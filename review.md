# MNIST-MSI Code Review

本文档整理当前项目中最需要优先处理的 5 个问题，并给出对应的修复方案与验证建议。

## 总体结论

当前项目的主要风险不在于模型能否跑通，而在于实验结论的可信度和实现细节的正确性：

- 神经网络的 early stopping 逻辑存在实现错误，当前代码并不会真正恢复到验证集最优权重。
- MPS/CUDA 训练和推理时间的统计方式不正确，README 中的耗时结论不可靠。
- KNN 和 Logistic Regression 的交叉验证过程存在数据泄漏，超参数搜索结果偏乐观。
- Logistic Regression 的实验设计被 KNN 的 PCA 选择结果绑死，模型对比不独立。
- CNN 数据增强按整个 batch 共享一组随机参数，增强效果弱于预期。

建议优先修复前两个 P1 问题，再处理实验设计和数据增强问题，最后补上最小化验证脚本或测试。

## 1. Early Stopping 无法恢复最佳权重

### 问题

`mnist_digit_recognition.ipynb` 中 `train_nn` 函数使用下面的方式保存最佳模型：

```python
best_model_state = model.state_dict().copy()
```

这只是对字典做浅拷贝，字典中的 tensor 仍然与模型参数共享底层数据。训练继续进行后，所谓的“最佳权重”会随着后续参数更新一起变化。最终执行：

```python
model.load_state_dict(best_model_state)
```

加载回来的其实是最后阶段的参数，而不是验证集表现最好的那一轮参数。

### 影响

- early stopping 逻辑名义上存在，实际上未生效。
- README 中神经网络最终精度可能对应的是最后一轮，而不是最佳验证轮次。
- 训练结果的可重复性与可信度下降。

### 修复方案

改为深拷贝权重：

```python
best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

加载时再直接：

```python
model.load_state_dict(best_model_state)
```

如果希望避免 CPU/GPU 之间来回切换，也可以使用：

```python
import copy
best_model_state = copy.deepcopy(model.state_dict())
```

### 建议验证

- 训练时打印最佳 epoch 与最终停止 epoch，确认二者可能不同。
- 训练完成后分别比较“停止时模型”和“恢复后模型”的验证集准确率，确认恢复动作确实生效。

## 2. MPS/CUDA 耗时统计方式错误

### 问题

当前训练和预测耗时都通过 `time.time()` 直接包裹前向、反向或预测过程统计，但对于 Apple MPS 和 CUDA，这些操作通常是异步提交的。代码没有在计时前后调用同步函数，因此测出来的时间会系统性偏小。

README 当前直接展示了这些时间，例如：

- CNN `~60s`
- MLP `~30s`
- KNN `~2s`
- Logistic Regression `~5s`

其中神经网络在 MPS 上的时间尤其容易失真。

### 影响

- README 中模型耗时对比不可信。
- “模型准确率 vs. 时间成本”的分析结论可能错误。
- 后续如果换到 CUDA 设备，误差会继续存在。

### 修复方案

增加统一同步函数：

```python
def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
```

在训练和预测计时前后调用：

```python
sync_device(device)
t_start = time.time()

# training / inference work

sync_device(device)
elapsed = time.time() - t_start
```

### 建议验证

- 在同一台机器上分别记录“同步前统计值”和“同步后统计值”，确认差异。
- README 中重新生成耗时结果，并注明硬件、PyTorch 版本与运行批大小。

## 3. KNN 与 LR 的交叉验证存在 PCA 数据泄漏

### 问题

当前流程先在整个子集上做 PCA，再把变换后的数据送入 `cross_val_score` 或 `GridSearchCV`：

```python
X_sub_pca = pca_temp.fit_transform(X_sub)
scores = cross_val_score(knn_temp, X_sub_pca, y_sub, cv=3, scoring='accuracy')
```

以及：

```python
X_sub_pca = pca_knn.transform(X_sub)
lr_grid.fit(X_sub_pca, y_sub)
```

这意味着每个验证折都看到了使用整份子集数据拟合出来的 PCA 基底，验证集信息被提前泄漏到训练过程里，导致 CV 分数偏乐观。

### 影响

- KNN 的 PCA 维度搜索结果不干净。
- KNN 和 LR 的超参数搜索结果都会偏乐观。
- README 中“最佳参数”和“CV accuracy”不够可信。

### 修复方案

将 PCA 放入 `Pipeline`，让每个 CV fold 只在训练折内部拟合 PCA。

KNN 示例：

```python
from sklearn.pipeline import Pipeline

knn_pipeline = Pipeline([
    ("pca", PCA()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "pca__n_components": [30, 50, 100],
    "knn__n_neighbors": [1, 3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}
```

Logistic Regression 同理：

```python
lr_pipeline = Pipeline([
    ("pca", PCA()),
    ("lr", LogisticRegression(max_iter=2000, random_state=42))
])
```

### 建议验证

- 重新运行 KNN 和 LR 的 CV 搜索，比较修复前后的最佳参数和准确率变化。
- 将 README 中关于“最佳 CV accuracy”的描述同步更新。

## 4. Logistic Regression 实验设计不独立

### 问题

当前 Logistic Regression 直接复用了 KNN 那一步得到的 `X_train_pca`、`X_test_pca` 和 `X_sub_pca`。而这些特征来自 KNN 所选出的 PCA 维度，不是为 LR 单独优化的。

这会让 LR 的最终表现依赖于 KNN 的搜索结果，而不是基于自身最优配置得到的结果。

### 影响

- README 中 Logistic Regression 的精度无法作为独立 baseline 解读。
- “LR 因线性边界不足所以只有约 89%” 这个结论依据不充分。
- 模型横向比较不公平。

### 修复方案

为 Logistic Regression 单独建立搜索流程，不要复用 KNN 的 PCA 结果。

建议两种方案二选一：

1. 完全不做 PCA，直接使用原始 784 维输入，并对 `C`、`solver`、`max_iter` 做搜索。
2. 使用 `Pipeline(PCA + LogisticRegression)`，把 `n_components` 也纳入 LR 自己的搜索空间。

更推荐方案 2，因为它同时兼顾计算效率和实验公平性。

例如：

```python
lr_param_grid = {
    "pca__n_components": [30, 50, 100, 150],
    "lr__C": [0.01, 0.1, 1.0, 10.0],
    "lr__solver": ["lbfgs", "saga"],
}
```

### 建议验证

- 分别记录 “原始特征 LR” 与 “PCA+LR” 的测试集表现。
- 在 README 中明确 LR 的实验设定，避免把 KNN 的预处理结果当作 LR 的默认输入。

## 5. CNN 数据增强按 batch 共享随机参数

### 问题

当前训练代码对整个 batch 一次性调用：

```python
X_img = X_batch.view(-1, 1, 28, 28)
X_img = cnn_augment(X_img)
X_batch = X_img.view(-1, 784)
```

在 `torchvision.transforms.RandomRotation` 和 `RandomAffine` 的这种写法下，整批样本会共享同一组随机旋转和平移参数，而不是每张图像各自独立采样。

这意味着一个 batch 内的所有图像会被“整体一起旋转/平移”，增强多样性明显不足。

### 影响

- 数据增强效果被削弱。
- CNN 的泛化收益低于预期。
- README 中“通过旋转/平移增强提升到 99.61%”这一因果链条证据不足。

### 修复方案

将增强改为按样本执行。常见做法有两种：

1. 自定义 `Dataset`，在 `__getitem__` 中把单张图像 reshape 成 `1x28x28` 后再执行 transform。
2. 保持当前 `TensorDataset` 结构，但在训练循环中逐样本应用 transform：

```python
X_img = X_batch.view(-1, 1, 28, 28)
X_img = torch.stack([cnn_augment(img) for img in X_img])
X_batch = X_img.view(-1, 784)
```

如果后续改成标准 CNN 输入形式，建议直接让数据集输出 `1x28x28`，避免每轮在训练循环中手动 reshape。

### 建议验证

- 对同一张图复制多份组成一个 batch，检查增强后每张图是否仍完全相同。
- 比较修复前后的验证集准确率与训练曲线，确认增强多样性增加后是否带来收益。

## 推荐修复顺序

1. 修复 early stopping 的权重保存方式。
2. 修复 MPS/CUDA 的计时逻辑并更新 README。
3. 将 KNN 和 LR 改为基于 `Pipeline` 的无泄漏交叉验证。
4. 为 LR 建立独立搜索流程。
5. 将 CNN 数据增强改成按样本执行。

## 建议补充的最小验证

当前仓库没有自动化测试，建议至少补充以下最小验证脚本或 notebook 检查单：

- 验证 early stopping 恢复的确是最佳权重。
- 验证 GPU/MPS 计时已做同步。
- 验证 KNN/LR 的 CV 流程中 PCA 在 fold 内拟合。
- 验证 CNN 增强对同 batch 相同样本会产生不同结果。

这样后续即使继续调整模型结构，也能避免这些基础问题再次回归。
