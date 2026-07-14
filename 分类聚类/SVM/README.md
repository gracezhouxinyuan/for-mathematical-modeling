# SVM（支持向量机）（2026-07-14）

```text
svm/
├── readme/
│   └── README.md
├── 代码/
│   ├── svm_baseline.py
│   ├── svm_improved_rbf_grid.py
│   └── plot_svm_figures.py
├── figure/
└── dataset/
    ├── svm_demo_data.csv
    ├── dataset_sources.md
    ├── results_baseline_metrics.csv
    └── results_improved_metrics.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：SVM（支持向量机）
- 核心公式：
  - 分类间隔最大化：`max 2 / ||w||`
  - 约束：`y_i(w^T x_i + b) >= 1`
  - 软间隔：`min 1/2||w||^2 + C sum xi_i`
  - 核函数：`K(x_i, x_j)` 将非线性问题映射到高维空间
- 适用条件：
  - 中小样本分类
  - 特征维度较高
  - 决策边界需要清晰解释
- 常见失败场景：
  - 样本量很大导致训练慢
  - 参数 `C` 与 `gamma` 未调优
  - 特征未标准化导致间隔失真
- 算法的改良算法

| 改良方法     | 如何改良                      | 相比原版的优点 | 相比原版的缺点         | 适用场景/数据集              |
| ------------ | ----------------------------- | -------------- | ---------------------- | ---------------------------- |
| 线性 SVM     | 使用线性核寻找最大间隔超平面  | 速度快、解释强 | 无法刻画复杂非线性边界 | 高维稀疏特征、文本分类       |
| RBF 核 SVM   | 用高斯核映射非线性边界        | 非线性表达强   | 参数敏感、训练较慢     | 小中样本非线性分类           |
| 网格搜索 SVM | 交叉验证选择 `C/gamma/kernel` | 泛化更稳       | 计算成本增加           | 国赛中需要稳健对比的分类问题 |
| 类别权重 SVM | 给少数类更高惩罚权重          | 提升少数类召回 | 可能牺牲总体准确率     | 不平衡分类、风险识别         |

**<u>使用建议</u>**

1. 特征标准化后再训练 SVM。
2. 线性可分或高维稀疏数据先用线性核。
3. 非线性边界明显时使用 RBF 核并做交叉验证。

### B. 代码

- baseline 文件：`代码/svm_baseline.py`
- 改良文件：`代码/svm_improved_rbf_grid.py`
- 可视化文件：`代码/plot_svm_figures.py`
- 可复现性：`Y`（固定随机种子与固定流程）

### C. 实验

- 指标：accuracy、F1、AUC
- 对比对象：linear SVM vs RBF GridSearch SVM
- 结论目标：说明核函数和参数搜索对非线性分类边界的提升。

### D. 论文

- 方法段：

> 本文针对非线性二分类问题构建支持向量机模型。基础模型采用线性核函数，通过最大化分类间隔获得决策超平面，具有结构清晰、泛化能力较强的优点。考虑到实际数据可能存在非线性边界，进一步引入 RBF 核函数，并通过交叉验证搜索惩罚参数和核参数，从而提升模型对复杂边界的表达能力。实验结果表明，改良后的核 SVM 在 F1 值和 AUC 指标上优于线性 SVM，适合国赛中的小中样本分类、风险识别和模式判别任务。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 svm/代码/svm_baseline.py
python3 svm/代码/svm_improved_rbf_grid.py
python3 svm/代码/plot_svm_figures.py
```

运行后你会得到：

- 终端输出：模型核心指标与部分结果预览
- 文件输出：
- `svm/dataset/results_baseline_metrics.csv`
- `svm/dataset/results_improved_metrics.csv`
- `svm/figure/fig_01_svm_comparison.png`

---

## 3. 数据集说明

本包使用 `make_moons` 合成非线性二分类数据，主要用于展示线性 SVM 与 RBF 核 SVM 的差异。真实数据建议优先来自题目附件、UCI 或政府公开数据。

官方公开来源：

- scikit-learn SVC 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>
- SVM 经典概念：<https://en.wikipedia.org/wiki/Support_vector_machine>
- UCI Machine Learning Repository：<https://archive.ics.uci.edu/>
