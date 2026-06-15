# NSGA-III（2026-06-15）

```text
nsga_iii/
├── readme/
│   └── README.md
├── 代码/
│   ├── nsga_iii_baseline.py
│   ├── nsga_iii_improved.py
│   └── plot_nsga_iii_figures.py
├── figure/
└── dataset/
    ├── nsga_iii_problem.csv
    ├── dataset_sources.md
    ├── results_baseline_nsga3.csv
    └── results_improved_nsga3.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：NSGA-III
- 核心思想：在 NSGA-II 基础上加入参考点/参考方向机制，改善多目标高维 Pareto 前沿的覆盖。
- 核心公式：
  - 仍然使用非支配排序筛选前沿
  - 参考点关联：把个体映射到最近参考方向
  - 用 niching 机制维持各参考点的解分布
- 适用条件：三目标及以上、多目标前沿更复杂、希望保持前沿分布均匀。
- 常见失败场景：参考点设置不合适、目标归一化不稳定、极端点覆盖不足。

### NSGA-III 的改良算法

| 改良方法     | 如何改良                              | 相比原版的优点         | 相比原版的缺点       | 适用场景/数据集      |
| ------------ | ------------------------------------- | ---------------------- | -------------------- | -------------------- |
| NSGA-II      | 用拥挤距离维护多样性                  | 实现更简单，适合双目标 | 高维前沿覆盖可能不均 | 双目标或低维多目标   |
| NSGA-III     | 引入参考方向和 niching                | 更适合三目标及以上     | 实现更复杂           | 高维 Pareto 前沿     |
| 自适应参考点 | 根据前沿分布动态调整 reference points | 覆盖更均匀             | 需要额外策略设计     | 目标维度高、分布不均 |
| 混合局部搜索 | 参考点选择后再做局部精修              | 前沿质量更高           | 代码更长             | 论文对解质量要求高时 |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/nsga_iii_baseline.py`
- 改良文件：`代码/nsga_iii_improved.py`
- 可视化文件：`代码/plot_nsga_iii_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对三目标多目标优化问题构建 NSGA-III 模型。首先使用 NSGA-II 作为基线得到初始 Pareto 解集，再引入参考点和 niching 机制，对高维前沿上的候选解进行关联与分配，从而提升解集分布均匀性。与仅依赖拥挤距离的 NSGA-II 相比，NSGA-III 更适合三目标及以上的场景，能够在更复杂的 Pareto 前沿上保持覆盖和多样性。实验表明，改良模型在三维目标空间中的前沿均匀性更好，适合国赛中的多目标资源配置、方案折中与高维权衡问题。

- 图表编号与说明：
  - `fig_01_nsga_iii_pipeline.png`：NSGA-III 数据处理与计算流程图
  - `fig_02_nsga_iii_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_nsga_iii_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：NSGA-III
- 核心思想：在 NSGA-II 基础上加入参考点/参考方向机制，改善多目标高维 Pareto 前沿的覆盖。
- 核心公式：
  - 仍然使用非支配排序筛选前沿
  - 参考点关联：把个体映射到最近参考方向
  - 用 niching 机制维持各参考点的解分布
- 适用条件：三目标及以上、多目标前沿更复杂、希望保持前沿分布均匀。
- 常见失败场景：参考点设置不合适、目标归一化不稳定、极端点覆盖不足。

### 数据集选择建议

适合三目标及以上的多目标优化，尤其适合需要均匀覆盖 Pareto 前沿的情形。

---

## 3. 运行方式

```bash
python3 nsga_iii/代码/nsga_iii_baseline.py
python3 nsga_iii/代码/nsga_iii_improved.py
python3 nsga_iii/代码/plot_nsga_iii_figures.py
```

---

## 4. 参考来源

- NSGA-III 原始论文 DOI：<https://doi.org/10.1109/TEVC.2013.2281535>
- 多目标优化基础：<https://en.wikipedia.org/wiki/Multi-objective_optimization>
- 参考方向方法：<https://en.wikipedia.org/wiki/Non-dominated_sorting_genetic_algorithm>
