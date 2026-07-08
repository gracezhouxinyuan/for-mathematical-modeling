# KNN（2026-06-06）

```text
knn/
├── readme/
│   └── README.md
├── 代码/
│   ├── knn_baseline.py
│   ├── knn_improved_weighted_metric.py
│   ├── knn_radius_neighbors.py
│   └── plot_knn_figures.py
├── figure/
└── dataset/
    ├── knn_demo_data.csv
    ├── knn_sources.md
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：KNN（K-Nearest Neighbors）
- 核心思想：一个样本的类别/数值由其最近的 k 个邻居决定
- 核心公式：
  - 距离：欧氏、曼哈顿、余弦等
  - 分类：多数投票
  - 回归：邻居均值或加权均值
- 适用条件：
  - 数据量中小
  - 特征经过标准化
  - 局部结构有意义
- 常见失败场景：
  - 未标准化导致距离失真
  - 高维下“距离集中”问题
  - k 取值不当，过拟合或欠拟合

### KNN 的改良算法

| 改良方法          | 如何改良                                | 相比原版的优点                   | 相比原版的缺点                 | 适用场景/数据集                      |
| ----------------- | --------------------------------------- | -------------------------------- | ------------------------------ | ------------------------------------ |
| 距离加权 KNN      | 邻居越近权重越大                        | 近邻影响更大，通常比均匀投票更稳 | 对噪声点更敏感一些             | 分类边界不规则、局部密度有差异的数据 |
| 自适应 KNN        | 不固定 k，按局部密度或验证集调整        | 更灵活，缓解 k 选择困难          | 实现更复杂，需要调参           | 结构复杂、局部密度不均的数据         |
| Radius Neighbors  | 用半径而不是固定 k 取邻居               | 对密度变化场景更自然             | 半径选取困难，稀疏区可能无邻居 | 非均匀密度数据、异常检测             |
| 近似最近邻（ANN） | 用 KDTree/BallTree/LSH/向量索引加速搜索 | 大数据下检索更快                 | 可能是近似而非精确最近邻       | 海量样本、实时推荐、检索系统         |
| 组合/原型选择 KNN | 先压缩样本或选代表点，再做 KNN          | 减少存储和查询成本               | 可能损失部分精度               | 大样本、实时分类、边缘设备           |

**使用建议**

1. 如果样本量不大，KNN 是非常好用的基线模型。
2. 如果边界复杂，先试距离加权 KNN。
3. 如果不知道固定 k 是否合理，考虑半径邻居或自适应 KNN。
4. 如果样本很多，优先考虑 ANN 或原型压缩。

### B. 代码

- baseline 文件：`代码/knn_baseline.py`
- 改良文件：`代码/knn_improved_weighted_metric.py`
- 进一步改良文件：`代码/knn_radius_neighbors.py`
- 可视化文件：`代码/plot_knn_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：准确率、F1、召回率、混淆矩阵
- 对比对象：
  - baseline KNN（均匀投票）
  - improved weighted KNN
  - radius neighbors
- 结论目标：证明距离加权与半径邻域在复杂边界数据上更稳

### D. 论文

> 本文针对样本分类问题构建 KNN 模型。首先对特征进行标准化处理，并计算样本之间的距离以确定最近邻集合。基础 KNN 采用固定 k 值和均匀投票进行类别判定，具有实现简单、可解释性强的优点，但在类别边界复杂、局部密度差异较大时容易受到远邻样本干扰。为提升分类稳定性，进一步引入距离加权机制，使得更近邻居对决策贡献更大，并结合半径邻域策略处理密度不均场景。实验结果表明，改良 KNN 在准确率、F1 值和边界样本判别能力上更优，适合国赛中的分类识别、局部模式判定与小中规模样本预测任务。

---

## 2. KNN 理论补充

KNN 的优点是“简单粗暴、解释性强”，缺点是“吃距离、吃尺度、吃计算”。所以它在国赛中最适合作为一个很强的基线模型，或者在小样本、局部结构明显的任务中直接使用。

### 优点

- 无需训练过程
- 可解释性强
- 非参数方法
- 对复杂边界有一定适应性

### 缺点

- 预测时计算量大
- 对特征尺度敏感
- 高维表现一般
- 受噪声和异常值影响

### 适合场景

- 小中样本分类
- 局部结构明显的问题
- 需要直观解释的任务

### 不太适合

- 海量样本实时分类
- 高维稀疏数据
- 特征完全不可比、未标准化的数据

---

## 3. 运行方式

```bash
python3 knn/代码/knn_baseline.py
python3 knn/代码/knn_improved_weighted_metric.py
python3 knn/代码/knn_radius_neighbors.py
python3 knn/代码/plot_knn_figures.py
```

---

## 4. 参考来源

- scikit-learn KNeighborsClassifier 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>
- scikit-learn KNeighborsRegressor 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>
- scikit-learn Nearest Neighbors 官方文档：<https://scikit-learn.org/stable/modules/neighbors.html>
- scikit-learn RadiusNeighborsClassifier 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html>
