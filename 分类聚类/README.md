# 🔍 分类聚类类算法库 (Classification & Clustering Methods)

## 概述 Overview

分类聚类算法用于**结构识别、分组分类**。包括无监督学习（聚类）和有监督学习（分类）两大类。

| 算法 | 任务类型 | 特点 | 应用场景 | 难度 |
|------|--------|------|--------|------|
| **KMeans** | 聚类 | 无监督，快速 | 初步分组，聚类 | ⭐ |
| **层次聚类** | 聚类 | 树状结构，多粒度 | 聚类树，分层分析 | ⭐⭐ |
| **SVM** | 分类 | 最大间隔，非线性 | 二分类，高维数据 | ⭐⭐⭐ |
| **DBSCAN** | 聚类 | 密度聚类，任意形状 | 异常点检测，不规则聚类 | ⭐⭐ |
| **随机森林** | 分类/回归 | 集成学习，高精度 | 特征重要性，分类 | ⭐⭐⭐ |
| **GMM** | 聚类 | 概率模型，软聚类 | 概率分布，不确定性 | ⭐⭐⭐ |
| **XGBoost** | 分类/回归 | 梯度提升，超强 | 竞赛分类，精准预测 | ⭐⭐⭐⭐ |

---

## 第一部分：聚类算法

## 1. KMeans（K均值聚类）

### 原理简介

将n个样本分成k个簇，**每个簇的样本距离簇心最近**。

**算法步骤：**
1. 随机初始化k个簇心
2. 将每个样本分配到最近的簇心
3. 更新簇心（所有样本的均值）
4. 重复2-3直到收敛

**目标函数：** 最小化簇内平方和(WSS)
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

### Python实现

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 生成样本数据
X = np.random.randn(100, 2) + np.array([[2, 2], [8, 8], [2, 8]])

# KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 绘图
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
           edgecolors='k', linewidths=2, label='Centers')
plt.legend()
plt.show()
```

### 自动选择最优K值

```python
from sklearn.metrics import silhouette_score

# 肘部法则（Elbow Method）
inertias = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# 绘图找"肘部"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K值')
axes[0].set_ylabel('簇内平方和')
axes[0].set_title('肘部法则')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'go-')
axes[1].set_xlabel('K值')
axes[1].set_ylabel('Silhouette系数')
axes[1].set_title('轮廓系数')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Silhouette系数越高越好
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"最优K值: {optimal_k}")
```

---

## 2. 层次聚类（Hierarchical Clustering）

### 原理

从单个样本开始，逐步合并最相似的两个簇，形成**树状聚类结果**。

**两种方法：**
1. **凝聚法（自底向上）**：从样本开始逐步合并
2. **分裂法（自顶向下）**：从全集开始逐步分割

**链接方式：**
- **单链接**：两簇间最小距离
- **完全链接**：两簇间最大距离
- **平均链接**：两簇间平均距离
- **Ward链接**：最小化簇内方差增加

### Python实现

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# 计算距离矩阵
distances = pdist(X, metric='euclidean')

# 凝聚聚类（Ward方法）
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(10, 6))
dendrogram(Z, leaf_font_size=10)
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.title('层次聚类树状图')
plt.axhline(y=15, c='r', linestyle='--', label='Cut threshold')
plt.legend()
plt.show()

# 根据阈值提取簇
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=15, criterion='distance')
```

---

## 3. DBSCAN（密度聚类）

### 原理简介

基于**样本密度**的聚类，能识别任意形状的簇并找出异常点。

**核心概念：**
- **核心点**：eps邻域内至少包含min_samples个样本
- **边界点**：不是核心点，但在某核心点的eps邻域内
- **噪声点**：既不是核心点也不是边界点

### Python实现

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 数据标准化
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 绘图
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
plt.title('DBSCAN聚类')
plt.show()

# 异常点（标签为-1）
n_noise = np.sum(labels == -1)
print(f"异常点数: {n_noise}")
```

### 参数调优

```python
from sklearn.neighbors import NearestNeighbors

# K距离图找eps值
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('样本排序')
plt.ylabel('第5邻近距离')
plt.title('K距离图（找eps的"肘部"）')
plt.grid(True)
plt.show()
```

---

## 第二部分：分类算法

## 4. SVM（支持向量机）

### 原理简介

在**最大间隔**的约束下找到最优分类超平面。

**二分类问题：**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_i \xi_i$$
$$s.t. \quad y_i(w^T\phi(x_i) + b) \geq 1 - \xi_i$$

其中：
- **w**: 超平面法向量
- **C**: 正则化参数（权衡准确度和间隔）
- **φ(x)**: 核函数映射

**常用核函数：**
- 线性核：$K(x_i, x_j) = x_i^T x_j$
- RBF核：$K(x_i, x_j) = \exp(-\gamma||x_i - x_j||^2)$
- 多项式核：$K(x_i, x_j) = (x_i^T x_j + c)^d$

### Python实现

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 线性SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
accuracy_linear = svm_linear.score(X_test, y_test)

# RBF核SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
accuracy_rbf = svm_rbf.score(X_test, y_test)

# 模型评估
y_pred = svm_rbf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优得分: {grid_search.best_score_:.4f}")
```

---

## 5. 随机森林（Random Forest）

### 原理

集成多个决策树的预测结果（投票/平均），具有**高精度和强鲁棒性**。

**优点：**
- 自动处理非线性关系
- 可获得特征重要性
- 并行化程度高
- 抗过拟合能力强

### Python实现

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# 随机森林分类
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# 特征重要性
importances = rf.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.title('特征重要性')
plt.tight_layout()
plt.show()

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 6. GMM（高斯混合模型）

### 原理

用**多个高斯分布的加权组合**来建模数据分布，是**概率聚类**方法。

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

其中：
- $\pi_k$: 第k个分量的权重
- $\mathcal{N}$: 高斯分布
- $\mu_k, \Sigma_k$: 第k个分量的均值和协方差

### Python实现

```python
from sklearn.mixture import GaussianMixture

# 拟合GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 聚类分配
labels = gmm.predict(X)

# 获得柔性聚类概率
proba = gmm.predict_proba(X)

# BIC与AIC准则选择最优K
bics = []
aics = []
K_range = range(1, 10)

for k in K_range:
    gmm_k = GaussianMixture(n_components=k, random_state=42)
    gmm_k.fit(X)
    bics.append(gmm_k.bic(X))
    aics.append(gmm_k.aic(X))

plt.figure()
plt.plot(K_range, bics, 'o-', label='BIC')
plt.plot(K_range, aics, 's-', label='AIC')
plt.xlabel('K（分量数）')
plt.ylabel('信息准则')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 7. XGBoost分类

XGBoost既可用于回归也可用于分类。详见[优化类README]或预测类README的XGBoost部分。

### 分类用例

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# XGBoost分类
params = {
    'objective': 'binary:logistic',  # 二分类
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(params, dtrain, num_boost_round=100)
y_pred = (model.predict(dtest) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
```

---

## 算法对比与选择

| 算法 | 有标签 | 数据量 | 速度 | 精准度 | 可解释性 | 国赛频率 |
|------|------|------|------|------|--------|--------|
| KMeans | ✗ | 中等 | 快 | 中 | 高 | ⭐⭐⭐ |
| 层次聚类 | ✗ | 小 | 中 | 中 | 高 | ⭐⭐ |
| DBSCAN | ✗ | 中等 | 中 | 中 | 中 | ⭐⭐ |
| SVM | ✓ | 中等 | 中 | 高 | 低 | ⭐⭐⭐ |
| 随机森林 | ✓ | 大 | 快 | 很高 | 中 | ⭐⭐⭐⭐ |
| GMM | ✗ | 中等 | 快 | 中 | 中 | ⭐ |
| XGBoost | ✓ | 大 | 快 | 很高 | 低 | ⭐⭐⭐⭐ |

---

## 评估指标

### 聚类指标
- **Silhouette系数**：-1~1，越接近1越好，衡量簇的紧凑度
- **Davies-Bouldin指数**：越小越好，衡量簇间分离度
- **Calinski-Harabasz指数**：越大越好

### 分类指标
- **准确率(Accuracy)**: $(TP + TN) / (TP + TN + FP + FN)$
- **精确率(Precision)**: $TP / (TP + FP)$
- **召回率(Recall)**: $TP / (TP + FN)$
- **F1-Score**: 精确率和召回率的调和平均
- **AUC-ROC**: 分类器排序能力

---

## 论文输出标准

1. **聚类结果可视化** → 散点图、热力图
2. **聚类有效性评估** → 表格形式的多个指标
3. **分类混淆矩阵** → 3×3或更大的矩阵
4. **特征重要性排序** → 柱状图
5. **ROC/PR曲线** → 多个模型对比
6. **参数敏感性分析** → 参数值 vs 性能曲线

---

## 常见陷阱

❌ **数据泄露**：参数调优用了测试集  
❌ **类别不平衡**：未采用加权或采样方法  
❌ **特征尺度差异大**：未进行标准化  
❌ **过拟合**：未使用交叉验证  

✓ **最佳实践**：标准化 → 交叉验证 → 超参数调优 → 多模型对比
