# 层次聚类（2026-07-09）

```text
hierarchical_clustering/
├── README.md
├── hierarchical_clustering_baseline.py
├── hierarchical_clustering_improved.py
├── plot_hierarchical_clustering_figures.py
├── dataset_sources.md
├── hierarchical_clustering_demo_data.csv
├── results_baseline_clusters.csv
├── results_improved_clusters.csv
└── results_metrics.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：层次聚类（Hierarchical Clustering）
- 核心思想：从单个样本开始逐步合并最相似的簇（凝聚法），或从全集开始逐步分割（分裂法），形成树状聚类结构
- 核心概念：
  - 凝聚法（自底向上）：每个样本初始为一个簇，逐步合并最近的两簇
  - 分裂法（自顶向下）：所有样本初始为一个簇，逐步分裂
  - 链接方式决定簇间距离的计算方法
- 链接方式：
  - 单链接（single）：两簇间最小距离 → 易产生链式效应
  - 完全链接（complete）：两簇间最大距离 → 簇紧凑但偏小球状
  - 平均链接（average）：两簇间平均距离 → 折中方案
  - Ward 链接：最小化合并后簇内方差增量 → 簇内方差最小，最常用
- 适用条件：
  - 需要展示数据的层次结构关系
  - 不预先指定簇数（可通过树状图截断确定）
  - 样本量中小规模（层次聚类计算复杂度 O(n²logn)）
- 常见失败场景：
  - 数据未标准化，大量纲特征主导聚类
  - 簇形状非凸或密度差异大，层次聚类效果差
  - 样本量过大导致计算不可行
  - 单链接产生链式效应，簇不紧凑

### 层次聚类的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
| -------- | -------- | -------------- | -------------- | --------------- |
| Ward 链接 | 以最小化簇内方差增量为合并准则 | 簇内方差最小，结果紧凑稳定 | 偏好球状簇 | 大多数数值型聚类任务 |
| 平均链接 | 以簇间平均距离为合并准则 | 对异常值较稳健，折中方案 | 不保证簇内紧凑 | 簇形状不规则的数据 |
| 完全链接 | 以簇间最大距离为合并准则 | 簇紧凑，直径可控 | 易被异常值影响 | 要求簇内紧凑的场景 |
| 层次聚类 + 距离度量优化 | 使用 Mahalanobis/余弦距离替代欧氏距离 | 适应不同数据结构 | 需要选择合适度量 | 文本/高维数据聚类 |
| BIRCH | 利用 CF树做层次聚类 | 适合大规模数据 | 对簇形状有假设 | 大规模数值数据 |
| 层次聚类 + 截断优化 | 用轮廓系数自动选截断阈值 | 自动确定簇数 | 增加计算成本 | 不确定簇数的探索性分析 |

**使用建议**

1. 先标准化数据，再计算距离矩阵。
2. 默认使用 Ward 链接 + 欧氏距离，这是最稳定的选择。
3. 如果 Ward 效果不好，尝试 average 或 complete 链接。
4. 通过树状图和轮廓系数共同确定最优截断高度。
5. 层次聚类特别适合需要展示层次关系的分析任务（如区域分层、物种分类）。

### B. 代码

- baseline 文件：`hierarchical_clustering_baseline.py`
- 改良文件：`hierarchical_clustering_improved.py`
- 可视化文件：`plot_hierarchical_clustering_figures.py`
- 可复现性：`Y`（固定随机种子与固定流程）

### C. 实验

- 指标：Silhouette 系数、Calinski-Harabasz 指数、Davies-Bouldin 指数
- 对比对象：
  - baseline：Ward 链接 + 固定 k=3
  - improved：多链接方式对比 + 轮廓系数自动选 k
- 结论目标：说明链接方式和自动簇数选择对层次聚类效果的影响。

### D. 论文

- 方法段：

> 本文针对样本分群问题采用层次聚类方法进行分析。首先对原始特征进行标准化处理，以消除不同量纲对距离计算的影响；随后采用凝聚式层次聚类，以 Ward 链接为合并准则，通过最小化簇内方差增量逐步合并样本。考虑到标准方法中簇数 k 需主观指定，进一步引入轮廓系数作为自动确定最优截断高度的准则，并与单链接、完全链接、平均链接等方法进行对比。实验结果表明，Ward 链接配合轮廓系数选 k 的方案在聚类紧凑度和分离度上表现最优，适用于国赛中的区域划分、样本分层和特征分组问题。

---

## 2. 层次聚类理论补充

层次聚类的核心优势在于**不需要预先指定簇数**，而是通过树状图（Dendrogram）展示数据的完整层次结构，研究者可以在任意高度截断以获得不同粒度的聚类结果。

### 优点

- 不需要预先指定簇数 k
- 树状图直观展示数据的层次结构
- 可在多个粒度上分析聚类结果
- 确定性算法（不含随机初始化，结果可复现）

### 缺点

- 计算复杂度高 O(n²logn)，不适合大规模数据
- 合并/分裂不可逆（一旦合并无法撤销）
- 对噪声和异常值敏感
- 不同链接方式对结果影响大

### 适合场景

- 需要展示层次关系的分类（生物分类、区域分层）
- 不确定簇数，需要探索性分析
- 中小规模样本聚类
- 需要可复现确定性结果的聚类

### 不太适合

- 大规模数据（用 KMeans 或 BIRCH）
- 非凸簇形状（用 DBSCAN）
- 高维稀疏数据

---

## 3. 运行方式

```bash
python3 hierarchical_clustering/hierarchical_clustering_baseline.py
python3 hierarchical_clustering/hierarchical_clustering_improved.py
python3 hierarchical_clustering/plot_hierarchical_clustering_figures.py
```

运行后你会得到：

- 终端输出：模型核心指标与部分结果预览
- 文件输出：
  - `results_baseline_clusters.csv`
  - `results_improved_clusters.csv`
  - `results_metrics.csv`

---

## 4. 参考来源

- SciPy hierarchy 官方文档：<https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>
- scikit-learn AgglomerativeClustering：<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
- 层次聚类原理：<https://en.wikipedia.org/wiki/Hierarchical_clustering>
- Ward 链接方法：<https://en.wikipedia.org/wiki/Ward%27s_method>
