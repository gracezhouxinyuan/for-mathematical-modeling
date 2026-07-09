# ODE（常微分方程）（2026-07-08）

```text
ode/
├── readme/
│   └── README.md
├── 代码/
│   ├── ode_baseline_logistic.py
│   ├── ode_improved_solver_compare.py
│   └── plot_ode_figures.py
├── figure/
└── dataset/
    ├── ode_logistic_observations.csv
    ├── dataset_sources.md
    ├── results_baseline_solution.csv
    └── results_solver_compare.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：ODE（Ordinary Differential Equation，常微分方程）
- 核心公式：
  - 一般形式：`dy/dt = f(t, y, θ)`
  - 初值问题：`y(t0)=y0`
  - Logistic 增长：`dN/dt = rN(1-N/K)`
  - 数值解法：Euler、RK4、Runge-Kutta、自适应步长法
- 适用条件：
  - 系统状态随时间连续变化
  - 变化规律能写成状态变量的导数
  - 需要预测、模拟或做机理解释
- 常见失败场景：
  - 参数没有实际意义或无法估计
  - 步长过大导致数值不稳定
  - 模型结构和真实机制不匹配
  - 忽略单位和时间尺度
- 算法的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| Euler 法 | 用一阶差分近似导数 | 最容易理解和手写 | 精度低、稳定性差 | 教学演示、粗略模拟 |
| RK4 | 每步计算四个斜率加权平均 | 精度高、实现不复杂 | 固定步长需手动设置 | 平滑 ODE、国赛常用模拟 |
| 自适应 Runge-Kutta | 根据误差自动调整步长 | 稳定、省心、精度可控 | 原理解释比 RK4 复杂 | 刚性不强的实际建模 |
| 刚性求解器 BDF/Radau | 用隐式方法处理刚性系统 | 稳定性更好 | 计算更重 | 快慢变量差异大的系统 |
| ODE + 参数估计 | 用观测数据反推参数 | 更贴近真实数据 | 需要优化和误差假设 | 生长、传播、药物动力学 |

**<u>使用建议</u>**

1. 先写清楚状态变量、单位、初值和参数含义。
2. 普通连续系统优先用 `solve_ivp` 的 `RK45`。
3. 如果曲线剧烈震荡或求解失败，检查步长、刚性和参数量纲。
4. 写论文时要配一张“模型结构图 + 时间演化曲线”。

### B. 代码

- baseline 文件：`代码/ode_baseline_logistic.py`
- 改良文件：`代码/ode_improved_solver_compare.py`
- 可视化文件：`代码/plot_ode_figures.py`
- 可复现性：`Y`（固定参数、固定时间网格）

### C. 实验

- 指标：最终状态、峰值、求解时间、与观测数据的 RMSE
- 对比对象：RK45 基础求解 vs RK23 / DOP853 / Radau 多求解器对比
- 结论目标：说明不同求解器在精度、稳定性和速度上的差异。

### D. 论文

- 方法段：

> 本文针对连续时间动态演化问题构建常微分方程模型。首先依据系统机理定义状态变量，并建立状态变量关于时间的导数方程；随后给定初始条件和参数，利用数值积分方法求解系统轨迹。以 Logistic 增长模型为例，模型通过增长率和环境容量刻画种群由快速增长到逐渐饱和的过程。考虑到不同数值求解器在精度和稳定性上存在差异，本文进一步比较 RK45、RK23、DOP853 和 Radau 等方法的求解结果与运行表现。实验结果表明，自适应 Runge-Kutta 方法能够在保证精度的同时保持较高效率，适合国赛中的人口增长、生态演化、药物浓度和传播动力学建模。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 ode/代码/ode_baseline_logistic.py
python3 ode/代码/ode_improved_solver_compare.py
python3 ode/代码/plot_ode_figures.py
```

运行后你会得到：

- 终端输出：ODE 关键指标和求解器对比
- 文件输出：
- `ode/dataset/results_baseline_solution.csv`
- `ode/dataset/results_solver_compare.csv`
- `ode/figure/fig_01_ode_logistic.png`

---

## 3. 数据集说明

本包使用合成 Logistic 增长观测数据，用于展示 ODE 数值求解与求解器对比流程。真实国赛数据可替换为人口、生态数量、药物浓度、温度变化、污染物浓度等连续时间观测数据。

官方公开来源：

- SciPy `solve_ivp` 官方文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
- 常微分方程概念：<https://en.wikipedia.org/wiki/Ordinary_differential_equation>
- Logistic function：<https://en.wikipedia.org/wiki/Logistic_function>
