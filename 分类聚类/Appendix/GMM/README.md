# GMM（2026-06-04）

```text
gmm/
├── readme/
│   └── README.md
├── 代码/
│   ├── gmm_baseline.py
│   ├── gmm_improved_bayesian.py
│   └── plot_gmm_figures.py
├── figure/
└── dataset/
    ├── gmm_demo_data.csv
    ├── gmm_sources.md
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：GMM（Gaussian Mixture Model，高斯混合模型）
- 核心思想：假设数据由若干个高斯分布混合而成，用 EM 算法估计每个成分的权重、均值和协方差
- 核心公式：
  - 混合密度：`p(x) = Σ_k π_k N(x | μ_k, Σ_k)`
  - E 步：计算每个样本属于各高斯成分的后验概率
  - M 步：更新 `π_k, μ_k, Σ_k`
- 适用条件：
  - 数据可能是椭球状簇
  - 需要软聚类结果（概率归属）
  - 簇间形状、方差不完全一致
- 常见失败场景：
  - 初始化不好导致 EM 陷入局部最优
  - 协方差矩阵退化或奇异
  - 真实分布与高斯假设偏差过大

### GMM 的改良算法

| 改良方法                       | 如何改良                                                   | 相比原版的优点                             | 相比原版的缺点           | 适用场景/数据集                        |
| ------------------------------ | ---------------------------------------------------------- | ------------------------------------------ | ------------------------ | -------------------------------------- |
| Bayesian GMM                   | 给混合权重、均值、协方差加先验，用变分推断或贝叶斯框架估计 | 可缓解过拟合，部分情况下可自动抑制无效成分 | 计算更复杂，解释成本更高 | 小样本、成分数不确定、需要更稳健的聚类 |
| 正则化协方差 GMM               | 在协方差上加 `reg_covar` 等正则项                          | 防止奇异矩阵，数值更稳定                   | 会对模型精度产生轻微偏置 | 高维数据、样本稀疏、协方差退化风险高   |
| Diagonal / Tied Covariance GMM | 约束协方差为对角或共享协方差                               | 参数更少，收敛更稳                         | 形状表达能力下降         | 高维、样本少、希望减少参数的场景       |
| 多初始化/多次重启 GMM          | 用不同初值多次训练，取最优似然                             | 降低局部最优风险                           | 计算时间更长             | 聚类边界复杂、对初始化敏感的数据       |
| GMM + PCA / 降维预处理         | 先降维再拟合混合模型                                       | 更稳、更快，适合高维数据                   | 可能损失信息             | 高维样本、特征相关性强的数据           |

**使用建议**

1. 如果你要软聚类或概率归属，优先 GMM。
2. 如果出现协方差奇异，优先正则化或对角协方差。
3. 如果簇数不确定或小样本，优先 Bayesian GMM。
4. 如果高维且特征相关明显，可先 PCA 再 GMM。

### B. 代码

- baseline 文件：`代码/gmm_baseline.py`
- 改良文件：`代码/gmm_improved_bayesian.py`
- 可视化文件：`代码/plot_gmm_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：对数似然、BIC/AIC、聚类准确率（若有标签）、软归属概率分布
- 对比对象：
  - baseline GMM
  - improved Bayesian GMM
- 结论目标：证明贝叶斯/正则化版本在参数稳定性和成分选择上更稳

### D. 论文

> 本文针对存在椭球形簇和软归属需求的数据，采用高斯混合模型（GMM）进行聚类分析。模型假设样本由多个高斯分布加权混合生成，并通过 EM 算法迭代估计各成分的权重、均值与协方差。考虑到标准 GMM 对初始值和协方差奇异较为敏感，进一步引入 Bayesian GMM 与协方差正则化策略，以增强参数估计的稳定性并降低过拟合风险。实验结果表明，改良模型在对数似然、BIC/AIC 及聚类稳定性方面表现更优，适合国赛中软聚类、异常识别和概率分群问题。

---

## 2. GMM 理论补充

GMM 比 KMeans 更“像真实世界”，因为它不是简单地把样本硬塞进某个簇，而是给每个样本一个属于各簇的概率。这样它更适合边界模糊、椭球形、重叠簇。

### 优点

- 支持软聚类
- 可以表达不同簇的形状与方差
- 概率模型更适合不确定性分析

### 缺点

- 依赖高斯假设
- 对初始化敏感
- 可能出现协方差奇异

### 适合场景

- 椭球形簇
- 重叠簇
- 需要概率输出
- 需要异常概率评估

### 不太适合

- 强非高斯分布
- 结构极其复杂的非凸簇
- 大规模高维且没有降维预处理的数据

---

## 3. 运行方式

```bash
python3 gmm/代码/gmm_baseline.py
python3 gmm/代码/gmm_improved_bayesian.py
python3 gmm/代码/plot_gmm_figures.py
```

---

## 4. 参考来源

- scikit-learn GaussianMixture 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html>
- scikit-learn BayesianGaussianMixture 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html>
- EM 算法经典说明：<https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>
- Bayesian GMM 综述：<https://scikit-learn.org/stable/modules/mixture.html>
