# TOPSIS 专题包

> 面向数学建模国赛的 TOPSIS 评价模型专题。这个目录可以直接作为仓库中的一个独立模块使用。

## 目录结构

```text
topsis/
├── readme/
│   └── README.md
├── 代码/
│   ├── topsis_baseline.py
│   ├── topsis_entropy_weight.py
│   ├── topsis_gra_compare.py
│   └── plot_topsis_figures.py
├── figure/
└── dataset/
    ├── topsis_demo_data.csv
    ├── topsis_sources.md
    └── results_metrics.csv
```

## 今日任务（2026-05-21）

### A. 理论（45分钟）
- 学习算法：TOPSIS（逼近理想解排序法）
- 核心思想：方案越接近正理想解、越远离负理想解，综合评价越高
- 核心公式：
  - 标准化后得到评价矩阵 `Z`
  - 正理想解 `z+`、负理想解 `z-`
  - 距离 `D+`、`D-`
  - 贴近度 `C = D- / (D+ + D-)`
- 适用条件：
  - 多指标综合评价
  - 指标方向一致化后可比
  - 需要给出排序而非分类
- 常见失败场景：
  - 指标未正向化
  - 权重设置主观且无依据
  - 不同量纲未标准化

### TOPSIS 原理补充

TOPSIS 的本质是“距离排序”。它先把每个方案映射到统一量纲空间，再比较该方案与两个极端参考点的距离。

- 正理想解：每个指标都取最优值。
- 负理想解：每个指标都取最差值。
- 贴近度：越接近 1 越好，越接近 0 越差。

TOPSIS 的优势是解释直观、实现简单、适合国赛里快速做排序；缺点是对权重和标准化方式较敏感，且默认指标之间相互独立。

### 改良算法详解

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| 熵权 TOPSIS | 用信息熵根据指标波动性自动赋权 | 减少主观权重，结果更客观 | 对数据分布较敏感，可能放大噪声指标 | 指标多、权重难确定、评价型面板数据 |
| 灰色关联 TOPSIS | 在 TOPSIS 前或过程中引入灰色关联度 | 对“小样本、弱信息”更稳健 | 计算步骤更多，解释稍复杂 | 样本少、指标不完全独立、结构关系弱的数据 |
| 模糊 TOPSIS | 用模糊数/区间数表达不确定指标 | 可处理专家语言评价和不确定信息 | 实现复杂，论文写作要说明模糊化规则 | 专家打分、语言变量、不确定性高的评价任务 |
| 区间 TOPSIS | 将指标看成区间而非单点 | 能表达波动范围与不确定性 | 需要区间定义清晰，否则容易争议 | 风险区间、价格区间、预测上下界数据 |
| AHP-TOPSIS / ANP-TOPSIS | 先做层次权重融合，再做排序 | 主客观结合，理论更完整 | 流程更长，对专家判断依赖仍存在 | 国赛评价类、综合决策类、层级指标问题 |

### 使用建议

- 如果权重已知且想快速排序，用基础 TOPSIS。
- 如果权重不清楚，优先熵权 TOPSIS。
- 如果数据少且结构关系复杂，可用灰色关联 TOPSIS。
- 如果专家只能给模糊意见，用模糊 TOPSIS。

### B. 代码（90分钟）
- baseline 文件：`代码/topsis_baseline.py`
- 改良文件：`代码/topsis_entropy_weight.py`
- 进一步改良文件：`代码/topsis_gra_compare.py`
- 可视化文件：`代码/plot_topsis_figures.py`
- 可复现：`Y`

### C. 实验（60分钟）
- 指标：排名一致性、贴近度分布、权重敏感性
- 对比对象：
  - baseline TOPSIS
  - 熵权 TOPSIS
  - 灰色关联增强 TOPSIS
- 结论目标：证明改良版本比固定权重版本更稳健

### D. 论文沉淀（30分钟）
- 方法段（150-300字）：

> 针对多指标综合评价问题，本文构建 TOPSIS 模型进行方案排序。首先对各指标进行正向化与向量标准化，随后计算各方案到正理想解与负理想解的欧氏距离，并以贴近度作为综合评价得分。考虑到传统 TOPSIS 中主观赋权可能影响结果稳定性，进一步引入熵权法进行客观赋权，并构建熵权 TOPSIS 改进模型。同时，为增强评价结果对指标关联结构的刻画，引入灰色关联分析与 TOPSIS 的融合版本进行对比。实验结果表明，改良模型在排序稳定性与解释性方面更优，适合国赛中的评价类和方案优选类问题。

- 图表编号建议：
  - `fig_01_topsis_pipeline.png`
  - `fig_02_topsis_ranking_bar.png`
  - `fig_03_weight_sensitivity.png`
  - `results_metrics.csv`

## 相关改良算法

- 熵权 TOPSIS：用信息熵自动计算指标权重，适合指标权重未知或主观性强的场景。
- 灰色关联 TOPSIS：把灰色关联度与 TOPSIS 结合，适合样本少、结构信息弱的场景。
- 模糊 TOPSIS：将模糊数、直觉模糊数或区间数纳入评价，适合不确定指标。
- 区间 TOPSIS：适合指标本身是区间值而非单点值。
- AHP-TOPSIS / ANP-TOPSIS：先做主客观权重融合，再做排序。

### 数据集选择建议

- 高校综合评价、城市竞争力、企业绩效、项目优选：适合 TOPSIS 系列。
- 指标是数值型且“越大越好/越小越好”清晰时，基础 TOPSIS 最容易落地。
- 如果数据含专家语言打分或指标区间波动，则优先改良版。

## 官方/论文来源

- TOPSIS 概念来源：Hwang and Yoon 的经典 MCDM 体系，可参考 ScienceDirect 综述页  
  https://www.sciencedirect.com/topics/engineering/technique-for-order-preference-by-similarity-to-ideal-solution
- 熵权 TOPSIS 改良示例  
  https://www.sciencedirect.com/science/article/pii/S0957417420309209
- 灰色关联与 TOPSIS 融合示例  
  https://econpapers.repec.org/RePEc:hin:jnlmpe:8761681

## 运行方式

```bash
python3 topsis/代码/topsis_baseline.py
python3 topsis/代码/topsis_entropy_weight.py
python3 topsis/代码/topsis_gra_compare.py
python3 topsis/代码/plot_topsis_figures.py
```
