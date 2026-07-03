# 非线性最小二乘拟合（2026-06-10）

```text
nonlinear_least_squares/
├── readme/
│   └── README.md
├── 代码/
│   ├── nonlinear_least_squares_baseline.py
│   ├── nonlinear_least_squares_improved.py
│   └── plot_nonlinear_least_squares_figures.py
├── figure/
└── dataset/
    ├── nls_demo_data.csv
    ├── dataset_sources.md
    ├── results_baseline_fit.csv
    └── results_improved_fit.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：非线性最小二乘拟合
- 核心思想：通过最小化残差平方和来估计非线性模型参数。
- 核心公式：
  - `min_theta sum_i (y_i - f(x_i; theta))^2`
  - 常用求解器包括 `curve_fit`、`least_squares`、高斯牛顿和 LM 算法。
- 适用条件：参数少到中等、函数形式已知、想做曲线拟合或参数反演。
- 常见失败场景：初值不合理、参数耦合严重、异常值多、模型形式选错。

### 非线性最小二乘拟合 的改良算法

| 改良方法               | 如何改良                        | 相比原版的优点       | 相比原版的缺点         | 适用场景/数据集          |
| ---------------------- | ------------------------------- | -------------------- | ---------------------- | ------------------------ |
| 基础 curve_fit         | 给定初值后直接拟合              | 上手快、结果直观     | 对初值敏感             | 拟合形式已知且初值较靠谱 |
| 鲁棒损失 least_squares | 使用 soft_l1 / huber 等鲁棒损失 | 对异常值更稳         | 需要解释鲁棒损失含义   | 含异常点的实验数据       |
| 多初值重启             | 尝试多个初始参数并选最优        | 降低陷入局部最优概率 | 计算量增加             | 非线性较强、参数耦合明显 |
| 加约束拟合             | 对参数范围加上下界              | 避免不合理解         | 需要先知道参数物理范围 | 物理过程拟合、工程反演   |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/nonlinear_least_squares_baseline.py`
- 改良文件：`代码/nonlinear_least_squares_improved.py`
- 可视化文件：`代码/plot_nonlinear_least_squares_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对参数估计问题构建非线性最小二乘拟合模型。首先根据已知机理选择指数衰减形式，并通过最小化残差平方和估计模型参数。基础拟合方法实现简单，但在初值不稳定或含异常值的情况下容易出现收敛不佳的问题。为提升鲁棒性，进一步引入多初值重启与鲁棒损失函数，在多个起点之间选择代价更小的解，并降低异常点对参数估计的影响。实验结果表明，改良版在参数稳定性和拟合误差上更优，适合国赛中的曲线拟合、参数反演与经验公式构建任务。

- 图表编号与说明：
  - `fig_01_nonlinear_least_squares_pipeline.png`：非线性最小二乘拟合 数据处理与计算流程图
  - `fig_02_nonlinear_least_squares_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_nonlinear_least_squares_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：非线性最小二乘拟合
- 核心思想：通过最小化残差平方和来估计非线性模型参数。
- 核心公式：
  - `min_theta sum_i (y_i - f(x_i; theta))^2`
  - 常用求解器包括 `curve_fit`、`least_squares`、高斯牛顿和 LM 算法。
- 适用条件：参数少到中等、函数形式已知、想做曲线拟合或参数反演。
- 常见失败场景：初值不合理、参数耦合严重、异常值多、模型形式选错。

### 数据集选择建议

适合已知函数形式的曲线拟合、物理参数识别和经验关系建模。

---

## 3. 运行方式

```bash
python3 nonlinear_least_squares/代码/nonlinear_least_squares_baseline.py
python3 nonlinear_least_squares/代码/nonlinear_least_squares_improved.py
python3 nonlinear_least_squares/代码/plot_nonlinear_least_squares_figures.py
```

---

## 4. 参考来源

- SciPy curve_fit 文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>
- SciPy least_squares 文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>
- 最小二乘法基础：<https://en.wikipedia.org/wiki/Least_squares>
