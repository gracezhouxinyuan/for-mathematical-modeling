# AHP（2026-05-19）

```text
ahp/
├── readme/
│   └── README.md
├── 代码/
│   ├── ahp_baseline.py
│   ├── ahp_improved_cr_check.py
│   └── plot_ahp_figures.py
├── figure/
└── dataset/
    ├── ahp_criteria_matrix.csv
    ├── ahp_alternatives_scores.csv
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：AHP（层次分析法）
- 核心公式：
  - 判断矩阵：`A=(a_ij), a_ij>0, a_ji=1/a_ij`
  - 权重向量：`Aw=λ_max w`，取最大特征值对应特征向量并归一化
  - 一致性指标：`CI=(λ_max-n)/(n-1)`
  - 一致性比率：`CR=CI/RI`，通常要求 `CR<0.1`
- 适用条件：
  - 多指标决策
  - 指标有层级结构
  - 需要可解释决策
- 常见失败场景：
  - 矩阵主观性过强导致 `CR>0.1`
  - 指标重叠造成重复计权
  - 1-9 标度缺乏业务依据

### B. 代码

- baseline 文件：`代码/ahp_baseline.py`
- 改良文件：`代码/ahp_improved_cr_check.py`
- 可视化文件：`代码/plot_ahp_figures.py`
- 可复现性：`Y`（固定输入与固定流程）

### C. 实验

- 指标：CR、权重稳定性、最终排序
- 对比对象：
  - baseline（仅计算权重）
  - improved（CR检验 + 不一致矩阵自动修正）
- 结论目标：改良版在轻度不一致场景下更稳定

### D. 论文

- 方法段：

> 本文针对多指标综合决策问题构建层次分析法（AHP）模型。首先依据目标层、准则层与方案层建立层次结构，并采用1-9标度法构造两两比较判断矩阵。随后通过特征根法求取各层权重，并计算一致性比率 CR 进行有效性检验。当 CR<0.1 时接受判断矩阵，否则通过一致性修正后重新计算。最终将各层权重逐级合成得到方案综合得分，实现对候选方案的可解释排序。实验表明，加入一致性修正后模型在权重稳定性与排序鲁棒性上表现更优。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 ahp/代码/ahp_baseline.py
python3 ahp/代码/ahp_improved_cr_check.py
python3 ahp/代码/plot_ahp_figures.py
```

运行后你会得到：

- 终端输出：权重、CI、CR、方案得分与排序
- 文件输出：
  - `ahp/figure/fig_01_ahp_hierarchy.png`
  - `ahp/figure/fig_02_judgement_matrix_heatmap.png`
  - `ahp/figure/fig_03_weight_comparison.png`
  - `ahp/dataset/tab_01_judgement_matrix_and_cr.csv`
  - `ahp/dataset/tab_02_ahp_weights_and_ranking.csv`

---

## 3. 数据集说明

官方公开来源：

- 中国国家统计局数据查询平台（官方）
  - 链接：<https://data.stats.gov.cn/>
- UCI Machine Learning Repository（官方）
  - 链接：<https://archive.ics.uci.edu/>

