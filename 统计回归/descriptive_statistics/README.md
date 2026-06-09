# 描述统计（2026-06-09）

```text
descriptive_statistics/
├── readme/
│   └── README.md
├── 代码/
│   ├── descriptive_statistics_baseline.py
│   ├── descriptive_statistics_improved.py
│   └── plot_descriptive_statistics_figures.py
├── figure/
└── dataset/
    ├── descriptive_stats_demo.csv
    ├── dataset_sources.md
    ├── results_baseline_summary.csv
    └── results_robust_summary.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：描述统计
- 核心思想：用均值、中位数、标准差、四分位数、偏度、峰度等指标快速概括数据分布。
- 核心公式：
  - 均值 `x̄ = (1/n) sum_i x_i`
  - 方差 `s^2 = sum_i (x_i - x̄)^2 / (n-1)`
  - 四分位距 `IQR = Q3 - Q1`
  - 偏度/峰度刻画分布形状
- 适用条件：数据清洗前的探索性分析、结果展示、论文前置描述。
- 常见失败场景：直接用均值概括偏态数据、忽略缺失值、异常值导致统计失真。

### 描述统计 的改良算法

| 改良方法           | 如何改良                                  | 相比原版的优点         | 相比原版的缺点           | 适用场景/数据集              |
| ------------------ | ----------------------------------------- | ---------------------- | ------------------------ | ---------------------------- |
| 基础描述统计       | 直接输出均值、标准差、分位数等            | 简单直观、适合快速概览 | 对异常值和偏态敏感       | 初步 EDA、论文数据概况       |
| 稳健统计           | 用中位数、MAD、Winsorize 替代部分均值统计 | 更抗异常值，适合脏数据 | 解释时要说明稳健处理规则 | 存在极端值或明显偏态的数据   |
| Bootstrap 置信区间 | 对统计量重复抽样估计置信区间              | 能给出不确定性范围     | 计算量增加               | 样本量不大但希望量化波动     |
| 分组统计/加权统计  | 按类别或权重计算统计量                    | 更贴合业务场景         | 需要额外分组信息         | 分地区、分行业、分人群的统计 |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/descriptive_statistics_baseline.py`
- 改良文件：`代码/descriptive_statistics_improved.py`
- 可视化文件：`代码/plot_descriptive_statistics_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对表格型数据的探索性分析构建描述统计流程。首先对数据进行缺失值与异常值识别，再计算均值、标准差、分位数、偏度和峰度等统计量，用于快速刻画数据的集中趋势、离散程度和分布形态。考虑到普通均值统计对异常值较为敏感，进一步引入稳健统计方法，通过中位数、四分位距和 Winsorize 处理降低极端值影响。实验表明，稳健描述统计更适合国赛中的脏数据预处理和数据概况展示，可为后续建模提供更可信的基础信息。

- 图表编号与说明：
  - `fig_01_descriptive_statistics_pipeline.png`：描述统计 数据处理与计算流程图
  - `fig_02_descriptive_statistics_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_descriptive_statistics_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：描述统计
- 核心思想：用均值、中位数、标准差、四分位数、偏度、峰度等指标快速概括数据分布。
- 核心公式：
  - 均值 `x̄ = (1/n) sum_i x_i`
  - 方差 `s^2 = sum_i (x_i - x̄)^2 / (n-1)`
  - 四分位距 `IQR = Q3 - Q1`
  - 偏度/峰度刻画分布形状
- 适用条件：数据清洗前的探索性分析、结果展示、论文前置描述。
- 常见失败场景：直接用均值概括偏态数据、忽略缺失值、异常值导致统计失真。

### 数据集选择建议

适合所有建模任务的前置探索，尤其适合含缺失值、异常值、偏态分布的原始表格数据。

---

## 3. 运行方式

```bash
python3 descriptive_statistics/代码/descriptive_statistics_baseline.py
python3 descriptive_statistics/代码/descriptive_statistics_improved.py
python3 descriptive_statistics/代码/plot_descriptive_statistics_figures.py
```

---

## 4. 参考来源

- pandas 描述统计文档：<https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics>
- SciPy stats 文档：<https://docs.scipy.org/doc/scipy/reference/stats.html>
- 描述统计概念：<https://en.wikipedia.org/wiki/Descriptive_statistics>
