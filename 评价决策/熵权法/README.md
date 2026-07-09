# 熵权法（2026-07-09）

```text
entropy_weight/
├── readme/
│   └── README.md
├── 代码/
│   ├── entropy_weight_baseline.py
│   ├── entropy_weight_improved.py
│   └── plot_entropy_weight_figures.py
├── figure/
└── dataset/
    ├── entropy_demo_data.csv
    ├── dataset_sources.md
    ├── results_baseline_scores.csv
    ├── results_entropy_weight_scores.csv
    └── results_entropy_weights.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：熵权法
- 核心思想：用信息熵衡量指标离散度，离散越大说明区分度越强，权重越高。
- 核心公式：
  - 指标归一化后得到矩阵 `X`
  - `p_ij = x_ij / sum_i x_ij`
  - `e_j = -k * sum_i(p_ij ln p_ij),  k = 1/ln(n)`
  - `d_j = 1 - e_j`
  - `w_j = d_j / sum_j d_j`
- 适用条件：指标型数据、多指标评价、希望减少主观赋权影响。
- 常见失败场景：指标差异很小、异常值过多、未正向化或未标准化。

### 熵权法 的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| 等权法 | 把所有指标赋予相同权重 | 实现最简单、解释直接 | 忽略指标区分度 | 权重完全未知时的初步比较 |
| 熵权 + TOPSIS | 先用熵权客观赋权，再做 TOPSIS 排序 | 权重客观、排序清晰 | 对异常值敏感 | 国赛中常见的评价类问题 |
| CRITIC | 综合指标波动和相关性赋权 | 比纯熵权更能抑制冗余指标 | 计算量稍高 | 指标相关性较强的场景 |
| AHP + 熵权 | 主客观权重融合 | 兼顾专家经验和数据客观性 | 流程更长 | 需要论文表达更完整的综合决策问题 |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/entropy_weight_baseline.py`
- 改良文件：`代码/entropy_weight_improved.py`
- 可视化文件：`代码/plot_entropy_weight_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对多指标评价问题构建熵权法模型。首先对指标进行正向化和无量纲化处理，再利用各指标在样本中的离散程度计算信息熵，并据此得到客观权重。与等权法相比，熵权法能够让波动更明显、区分度更强的指标获得更高权重，从而减少主观赋权带来的偏差。实验中进一步将等权评分与熵权评分进行对比，结果表明熵权法在指标差异明显的场景下更具解释性和稳定性，适合国赛中的综合评价与方案排序任务。

- 图表编号与说明：
  - `fig_01_entropy_weight_pipeline.png`：熵权法 数据处理与计算流程图
  - `fig_02_entropy_weight_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_entropy_weight_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：熵权法
- 核心思想：用信息熵衡量指标离散度，离散越大说明区分度越强，权重越高。
- 核心公式：
  - 指标归一化后得到矩阵 `X`
  - `p_ij = x_ij / sum_i x_ij`
  - `e_j = -k * sum_i(p_ij ln p_ij),  k = 1/ln(n)`
  - `d_j = 1 - e_j`
  - `w_j = d_j / sum_j d_j`
- 适用条件：指标型数据、多指标评价、希望减少主观赋权影响。
- 常见失败场景：指标差异很小、异常值过多、未正向化或未标准化。

### 数据集选择建议

适合指标多、权重难以人工确定、且不同指标区分度差异明显的综合评价任务。

---

## 3. 运行方式

```bash
python3 entropy_weight/代码/entropy_weight_baseline.py
python3 entropy_weight/代码/entropy_weight_improved.py
python3 entropy_weight/代码/plot_entropy_weight_figures.py
```

---

## 4. 参考来源

- 信息熵基础：<https://en.wikipedia.org/wiki/Entropy_(information_theory)>
- pandas 官方文档：<https://pandas.pydata.org/docs/>
- NumPy 官方文档：<https://numpy.org/doc/>
