# Logistic Regression（2026-06-08）

```text
logistic_regression/
├── readme/
│   └── README.md
├── 代码/
│   ├── logistic_regression_baseline.py
│   ├── logistic_regression_improved.py
│   └── plot_logistic_regression_figures.py
├── figure/
└── dataset/
    ├── logistic_demo_data.csv
    ├── dataset_sources.md
    ├── results_baseline_metrics.csv
    ├── results_baseline_predictions.csv
    ├── results_improved_metrics.csv
    └── results_improved_predictions.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：Logistic Regression
- 核心思想：用 sigmoid 函数把线性组合映射为概率，再通过最大似然估计学习参数。
- 核心公式：
  - `p(y=1|x) = 1 / (1 + exp(-(w^T x + b)))`
  - `logit(p) = ln(p/(1-p)) = w^T x + b`
  - 损失常用对数似然 / 交叉熵
- 适用条件：二分类、概率解释需求强、希望模型可解释。
- 常见失败场景：强非线性边界、特征共线性严重、类别不平衡、异常值过多。

### Logistic Regression 的改良算法

| 改良方法              | 如何改良                             | 相比原版的优点       | 相比原版的缺点       | 适用场景/数据集              |
| --------------------- | ------------------------------------ | -------------------- | -------------------- | ---------------------------- |
| 基础逻辑回归          | 标准化后直接拟合 LogisticRegression  | 实现简单、可解释性强 | 对不平衡与阈值敏感   | 二分类基线、概率预测         |
| 类权重平衡 + 阈值优化 | 给少数类更大权重，并在验证集上选阈值 | 更适合不平衡数据     | 需要额外验证流程     | 风控、故障识别、违约判别     |
| L1 / Elastic Net      | 用稀疏正则做变量选择                 | 可减少冗余特征       | 调参更复杂           | 高维稀疏特征                 |
| 多项式/交互项扩展     | 引入交互特征提高非线性表达           | 增强拟合能力         | 解释性下降、易过拟合 | 边界略复杂但仍可线性化的问题 |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/logistic_regression_baseline.py`
- 改良文件：`代码/logistic_regression_improved.py`
- 可视化文件：`代码/plot_logistic_regression_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对二分类判定问题构建逻辑回归模型。首先对特征进行标准化处理，再通过线性组合与 sigmoid 函数将样本映射为类别概率，并用对数似然函数估计模型参数。基础模型具有实现简洁、可解释性强的优点，但在类别不平衡和阈值不固定的场景下容易出现召回率不足的问题。为此进一步引入类别权重平衡与阈值优化策略，在验证集上选择更合适的判定阈值，从而提升少数类识别能力。实验结果表明，改良版在准确率、F1 值和 AUC 上更稳定，适合国赛中的风险判定、二分类筛选和可解释概率预测任务。

- 图表编号与说明：
  - `fig_01_logistic_regression_pipeline.png`：Logistic Regression 数据处理与计算流程图
  - `fig_02_logistic_regression_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_logistic_regression_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：Logistic Regression
- 核心思想：用 sigmoid 函数把线性组合映射为概率，再通过最大似然估计学习参数。
- 核心公式：
  - `p(y=1|x) = 1 / (1 + exp(-(w^T x + b)))`
  - `logit(p) = ln(p/(1-p)) = w^T x + b`
  - 损失常用对数似然 / 交叉熵
- 适用条件：二分类、概率解释需求强、希望模型可解释。
- 常见失败场景：强非线性边界、特征共线性严重、类别不平衡、异常值过多。

### 数据集选择建议

适合二分类、概率预测、决策解释要求较高的场景，尤其适合类别不平衡的筛选类任务。

---

## 3. 运行方式

```bash
python3 logistic_regression/代码/logistic_regression_baseline.py
python3 logistic_regression/代码/logistic_regression_improved.py
python3 logistic_regression/代码/plot_logistic_regression_figures.py
```

---

## 4. 参考来源

- scikit-learn LogisticRegression 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- 逻辑回归基础：<https://en.wikipedia.org/wiki/Logistic_regression>
- 线性模型综述：<https://scikit-learn.org/stable/modules/linear_model.html>
