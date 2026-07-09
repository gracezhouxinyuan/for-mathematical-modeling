# SIR / SEIR 传染病模型（2026-07-08）

```text
sir_seir/
├── readme/
│   └── README.md
├── 代码/
│   ├── sir_baseline.py
│   ├── seir_improved.py
│   └── plot_sir_seir_figures.py
├── figure/
└── dataset/
    ├── epidemic_observations.csv
    ├── dataset_sources.md
    ├── results_sir.csv
    └── results_seir.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：SIR / SEIR 传染病动力学模型
- 核心公式：
  - SIR：`dS/dt=-βSI/N`，`dI/dt=βSI/N-γI`，`dR/dt=γI`
  - SEIR：`dS/dt=-βSI/N`，`dE/dt=βSI/N-σE`，`dI/dt=σE-γI`，`dR/dt=γI`
  - 基本再生数：`R0 = β/γ`
- 适用条件：
  - 人群可分为若干状态舱室
  - 传播过程有明确机理
  - 需要预测峰值、持续时间和干预影响
- 常见失败场景：
  - 参数没有依据
  - 人群混合假设过强
  - 忽略潜伏期、隔离、输入病例和时变传播率
  - 用累计数据直接拟合现存感染人数
- 算法的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| SIR | 将人群分为易感、感染、移除 | 结构简单、解释清楚 | 无潜伏期 | 快速疫情模拟 |
| SEIR | 增加潜伏者 `E` | 更符合有潜伏期疾病 | 参数更多 | 呼吸道传染病、潜伏期明显疾病 |
| SIRD | 增加死亡状态 `D` | 能刻画死亡风险 | 需要死亡率数据 | 公共卫生风险评估 |
| 时变 β 模型 | 让传播率随政策或行为变化 | 能模拟干预效果 | 需要分段假设 | 封控、疫苗、隔离政策分析 |
| 网络传播模型 | 用接触网络替代均匀混合 | 更细致 | 数据要求高 | 校园、社区、交通网络传播 |

**<u>使用建议</u>**

1. 先用 SIR 做基线，再用 SEIR 解释潜伏期。
2. 参数必须写清来源：题目、文献、估计或合理假设。
3. 论文里至少给出感染峰值、峰值时间和 `R0`。
4. 如果干预措施明显，考虑分段或时变传播率。

### B. 代码

- baseline 文件：`代码/sir_baseline.py`
- 改良文件：`代码/seir_improved.py`
- 可视化文件：`代码/plot_sir_seir_figures.py`
- 可复现性：`Y`（固定参数和时间网格）

### C. 实验

- 指标：感染峰值、峰值时间、最终感染规模、`R0`
- 对比对象：SIR vs SEIR
- 结论目标：说明加入潜伏期后，感染峰值和峰值时间会发生怎样的变化。

### D. 论文

- 方法段：

> 本文针对传染病传播过程构建 SIR 与 SEIR 舱室模型。首先将总人群划分为易感者、感染者和移除者，并通过传播率和恢复率描述不同状态之间的转移关系；随后考虑部分疾病存在潜伏期，进一步引入暴露者舱室，构建 SEIR 改良模型。通过数值积分求解各舱室随时间变化的轨迹，并计算感染峰值、峰值时间和基本再生数等指标。实验结果表明，相比基础 SIR 模型，SEIR 模型能够刻画潜伏期导致的传播滞后现象，更适合用于具有潜伏过程的疫情预测和干预评估。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 sir_seir/代码/sir_baseline.py
python3 sir_seir/代码/seir_improved.py
python3 sir_seir/代码/plot_sir_seir_figures.py
```

运行后你会得到：

- 终端输出：峰值、峰值时间、最终规模和 `R0`
- 文件输出：
- `sir_seir/dataset/results_sir.csv`
- `sir_seir/dataset/results_seir.csv`
- `sir_seir/figure/fig_01_sir_seir_compare.png`

---

## 3. 数据集说明

本包使用合成疫情观测序列，重点展示 SIR / SEIR 的建模流程。真实数据可使用题目附件、疾控中心公开数据、WHO 或 Our World in Data 等公开疫情数据。

官方公开来源：

- SIR model：<https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>
- SciPy `solve_ivp` 官方文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
- WHO 数据入口：<https://data.who.int/>
