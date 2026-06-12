# KMeans (2026-06-12)

## 目录结构

```text
kmeans/
├── readme/
│   └── README.md
├── 代码/
│   ├── kmeans_baseline.py
│   ├── kmeans_improved.py
│   ├── kmeans_bisecting_compare.py
│   └── plot_kmeans_figures.py
├── figure/
└── dataset/
    ├── kmeans_demo_data.csv
    ├── kmeans_sources.md
    └── results_metrics.csv
```

## 今日任务（2026-05-21）

### A. 理论（45分钟）

- 学习算法：KMeans 聚类
- 核心思想：最小化簇内平方和（inertia），让样本自动聚成 k 个簇
- 核心公式：
  - 目标函数 `J = Σ_i Σ_{x in C_i} ||x - μ_i||^2`
  - 迭代步骤：分配样本到最近中心，更新中心
- 适用条件：
  - 数值型样本聚类
  - 近似球状、方差相近簇
  - 需要解释簇中心
- 常见失败场景：
  - 变量未标准化
  - 初始化差导致局部最优
  - 选错 k 值

### KMeans 原理补充

KMeans 的核心是在“给定 k 个簇”的前提下，让每个样本归入最近中心，并不断更新中心直到收敛。

- 第一步：随机或按规则初始化聚类中心。
- 第二步：把样本分配到最近中心。
- 第三步：根据簇内样本均值更新中心。
- 第四步：重复直到中心不再变化或达到最大迭代次数。

它的优点是实现简单、速度快、结果直观；缺点是对初始中心、异常值和簇形状很敏感，不适合非凸、长条形或密度差异很大的数据。

### 改良算法详解

| 改良方法           | 如何改良                                 | 相比原版的优点                                 | 相比原版的缺点                   | 适用场景/数据集                                |
| ------------------ | ---------------------------------------- | ---------------------------------------------- | -------------------------------- | ---------------------------------------------- |
| k-means++          | 用概率方式优先选取彼此距离较远的初始中心 | 初始化更稳，通常更快收敛，局部最优风险更低     | 仍然需要预先指定 k               | 大多数数值型聚类任务，尤其是中心初值敏感的数据 |
| MiniBatchKMeans    | 每次用小批量样本更新中心                 | 速度快、内存占用低，适合大样本                 | 可能略牺牲精度，结果有一定随机性 | 海量样本、在线聚类、需要快速近似结果的数据     |
| BisectingKMeans    | 先把所有数据当成一个簇，再反复二分裂簇   | 常比普通 KMeans 更稳定，适合层次结构明显的数据 | 实现和解释比基础 KMeans 更复杂   | 大规模样本、簇结构层次明显、希望逐步分裂的任务 |
| Scalable k-means++ | 面向大规模数据的改良初始化               | 更适合分布式/大数据环境                        | 理论和实现复杂度更高             | 分布式计算、大规模数值数据                     |
| X-means            | 在聚类过程中自动估计最优簇数             | 缓解“k 取多少”的问题                           | 需要额外模型选择准则，计算更复杂 | 不知道簇数、希望自动选 k 的探索性分析          |

### 使用建议

- 如果只是普通数值聚类，先用 k-means++。
- 如果样本量很大，用 MiniBatchKMeans。
- 如果你怀疑数据有层次结构，用 BisectingKMeans。
- 如果不知道 k，先做肘部法/轮廓系数，再考虑 X-means 这类扩展思路。

### B. 代码（90分钟）

- baseline 文件：`代码/kmeans_baseline.py`
- 改良文件：`代码/kmeans_improved.py`
- 进一步改良文件：`代码/kmeans_bisecting_compare.py`
- 可视化文件：`代码/plot_kmeans_figures.py`
- 可复现：`Y`

### C. 实验（60分钟）

- 指标：inertia、silhouette、轮廓结构稳定性
- 对比对象：
  - random init KMeans
  - k-means++ KMeans
  - MiniBatchKMeans
  - BisectingKMeans
- 结论目标：证明初始化和增量聚类对稳定性/速度的提升

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对样本分群问题采用 KMeans 聚类方法进行分析。首先对原始特征进行标准化，以消除不同量纲对距离计算的影响；随后以簇内平方和最小为目标，通过迭代完成样本分配与聚类中心更新。考虑到标准 KMeans 对初始中心敏感，进一步引入 k-means++ 初始化以提高聚类稳定性，并与 MiniBatchKMeans 和 BisectingKMeans 进行对比。实验结果表明，改良初始化可显著降低局部最优风险，分裂式聚类与小批量聚类则在大样本场景下具有更好的效率表现。该框架适用于国赛中的样本分层、区域划分和异常识别问题。

- 图表编号建议：
  - `fig_01_kmeans_clusters.png`
  - `fig_02_inertia_k_comparison.png`
  - `fig_03_silhouette_comparison.png`
  - `results_metrics.csv`

## 相关改良算法

- k-means++：改进初始化，降低随机初始中心带来的不稳定。
- MiniBatchKMeans：小批量迭代，适合大样本数据。
- BisectingKMeans：二分裂式聚类，常用于大规模样本或层次结构明显的数据。
- Scalable k-means++：面向大规模场景的可扩展初始化。
- X-means：自动估计簇数的扩展思路，适合不知道 k 的情形。

### 数据集选择建议

- 场景更适合数值型、可标准化、簇形状相对规则的数据。
- 如果有类别型变量，先做编码或改用其他混合聚类方法。
- 如果簇密度差异很大或形状非凸，KMeans 往往不是最优选择。

## 运行方式

```bash
python3 kmeans/代码/kmeans_baseline.py
python3 kmeans/代码/kmeans_improved.py
python3 kmeans/代码/kmeans_bisecting_compare.py
python3 kmeans/代码/plot_kmeans_figures.py
```

## 参考来源

- scikit-learn KMeans 官方文档  
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn MiniBatchKMeans 官方文档  
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
- scikit-learn BisectingKMeans 官方文档  
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
- k-means++ 原始论文（Stanford）  
  https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- Scalable k-means++ 原始论文（Stanford）  
  https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
