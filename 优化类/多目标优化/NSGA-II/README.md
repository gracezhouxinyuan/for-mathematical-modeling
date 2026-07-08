# NSGA-II（2026-06-13）

```text
nsga_ii/
├── readme/
│   └── README.md
├── 代码/
│   ├── nsga_ii_baseline.py
│   ├── nsga_ii_improved.py
│   └── plot_nsga_ii_figures.py
├── figure/
└── dataset/
    ├── nsga_ii_problem.csv
    ├── dataset_sources.md
    ├── results_baseline_nsga2.csv
    └── results_improved_nsga2.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：NSGA-II
- 核心思想：通过非支配排序和拥挤距离在多目标空间中逼近 Pareto 前沿。
- 核心公式：
  - 非支配排序：用支配关系给个体分层
  - 拥挤距离：衡量同一前沿中个体的稀疏程度
  - 精英保留：合并父代和子代后筛选最优层
- 适用条件：多个互相冲突的目标，需要给出 Pareto 解集而不是单个最优解。
- 常见失败场景：前沿覆盖不足、约束处理不当、种群多样性下降。

### NSGA-II 的改良算法

| 改良方法     | 如何改良                         | 相比原版的优点         | 相比原版的缺点             | 适用场景/数据集       |
| ------------ | -------------------------------- | ---------------------- | -------------------------- | --------------------- |
| 加权和 GA    | 把多目标压成一个加权标量再优化   | 实现简单               | 只能得到少数解，前沿覆盖差 | 快速基线对比          |
| NSGA-II      | 非支配排序 + 拥挤距离 + 精英保留 | 能输出一条 Pareto 前沿 | 参数仍需调节               | 双目标/多目标组合优化 |
| 约束 NSGA-II | 把约束可行性纳入排序规则         | 更适合真实问题         | 实现更复杂                 | 有复杂约束的工程优化  |
| 混合局部搜索 | NSGA-II 后接局部精修             | 前沿质量更高           | 代码更长                   | 论文需要更高解质量时  |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/nsga_ii_baseline.py`
- 改良文件：`代码/nsga_ii_improved.py`
- 可视化文件：`代码/plot_nsga_ii_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对双目标组合优化问题构建多目标进化算法。首先通过加权和遗传搜索得到一个单点基线结果，再引入非支配排序与拥挤距离构建 NSGA-II 模型，以精英保留机制维护 Pareto 前沿的多样性。与单目标加权法相比，NSGA-II 不再把多个目标强行压缩为一个标量，而是输出一组互不支配的候选解，便于决策者根据偏好选择方案。实验表明，改良模型在前沿覆盖范围和解集多样性上明显更优，适合国赛中的多目标调度、资源分配和方案权衡问题。

- 图表编号与说明：
  - `fig_01_nsga_ii_pipeline.png`：NSGA-II 数据处理与计算流程图
  - `fig_02_nsga_ii_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_nsga_ii_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：NSGA-II
- 核心思想：通过非支配排序和拥挤距离在多目标空间中逼近 Pareto 前沿。
- 核心公式：
  - 非支配排序：用支配关系给个体分层
  - 拥挤距离：衡量同一前沿中个体的稀疏程度
  - 精英保留：合并父代和子代后筛选最优层
- 适用条件：多个互相冲突的目标，需要给出 Pareto 解集而不是单个最优解。
- 常见失败场景：前沿覆盖不足、约束处理不当、种群多样性下降。

### 数据集选择建议

适合需要输出一组折中解的双目标或多目标优化问题。

---

## 3. 运行方式

```bash
python3 nsga_ii/代码/nsga_ii_baseline.py
python3 nsga_ii/代码/nsga_ii_improved.py
python3 nsga_ii/代码/plot_nsga_ii_figures.py
```

---

## 4. 参考来源

- NSGA-II 原始论文 DOI：<https://doi.org/10.1109/4235.996017>
- 多目标优化基础：<https://en.wikipedia.org/wiki/Multi-objective_optimization>
- Pareto 效率：<https://en.wikipedia.org/wiki/Pareto_efficiency>
