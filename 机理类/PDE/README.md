# PDE（偏微分方程）（2026-07-08）

```text
pde/
├── readme/
│   └── README.md
├── 代码/
│   ├── pde_baseline_heat_explicit.py
│   ├── pde_improved_crank_nicolson.py
│   └── plot_pde_figures.py
├── figure/
└── dataset/
    ├── heat_equation_config.csv
    ├── dataset_sources.md
    ├── results_heat_explicit.csv
    └── results_heat_crank_nicolson.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：PDE（Partial Differential Equation，偏微分方程）
- 核心公式：
  - 一维热传导方程：`∂u/∂t = α ∂²u/∂x²`
  - 显式差分：`u_i^{n+1}=u_i^n+r(u_{i+1}^n-2u_i^n+u_{i-1}^n)`
  - 稳定条件：`r = αΔt/Δx² <= 1/2`
  - Crank-Nicolson：时间方向半隐式，兼顾稳定性和精度
- 适用条件：
  - 状态同时随时间和空间变化
  - 需要模拟扩散、传热、波动、流体或污染传播
  - 边界条件和初值能够明确给出
- 常见失败场景：
  - 网格过粗导致数值误差大
  - 显式格式不满足稳定条件
  - 边界条件设置不合理
  - 参数单位和空间尺度不统一
- 算法的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| 显式有限差分 | 用当前时刻值直接计算下一时刻 | 实现简单、直观 | 稳定条件严格 | 教学演示、短时间扩散 |
| 隐式有限差分 | 每一步求解线性方程组 | 稳定性更强 | 每步计算更重 | 大步长扩散问题 |
| Crank-Nicolson | 显式和隐式折中 | 二阶精度、稳定性好 | 实现比显式复杂 | 热传导、污染扩散 |
| 有限元 FEM | 将区域划分为单元并构造弱形式 | 适合复杂几何 | 理论和代码复杂 | 工程结构、复杂边界 |
| 有限体积 FVM | 保持守恒量离散守恒 | 守恒性质好 | 实现更专业 | 流体、污染物输运 |

**<u>使用建议</u>**

1. 国赛中如果只需要展示机理，优先用有限差分。
2. 显式格式必须检查稳定条件。
3. 如果要用较大时间步长，考虑隐式或 Crank-Nicolson。
4. 论文里要写清初值、边界条件、网格步长和稳定性条件。

### B. 代码

- baseline 文件：`代码/pde_baseline_heat_explicit.py`
- 改良文件：`代码/pde_improved_crank_nicolson.py`
- 可视化文件：`代码/plot_pde_figures.py`
- 可复现性：`Y`（固定网格和参数）

### C. 实验

- 指标：温度场最大值、能量衰减、稳定性、最终空间分布
- 对比对象：显式有限差分 vs Crank-Nicolson
- 结论目标：说明半隐式格式在稳定性和精度上的优势。

### D. 论文

- 方法段：

> 本文针对随时间和空间共同变化的扩散过程构建偏微分方程模型。以一维热传导方程为例，首先给定初始温度分布和边界条件，并利用有限差分方法对时间和空间导数进行离散化。基础模型采用显式差分格式，结构简单且便于解释，但需要满足稳定性条件。为提高数值稳定性和精度，进一步引入 Crank-Nicolson 半隐式格式，在每个时间步求解线性方程组得到下一时刻温度场。实验结果表明，改良格式在较长时间模拟中更稳定，适合国赛中的热传导、污染扩散和空间动态演化问题。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 pde/代码/pde_baseline_heat_explicit.py
python3 pde/代码/pde_improved_crank_nicolson.py
python3 pde/代码/plot_pde_figures.py
```

运行后你会得到：

- 终端输出：网格参数、稳定系数和最终状态摘要
- 文件输出：
- `pde/dataset/results_heat_explicit.csv`
- `pde/dataset/results_heat_crank_nicolson.csv`
- `pde/figure/fig_01_pde_heat_equation.png`

---

## 3. 数据集说明

本包使用一维热传导方程的合成初值和边界条件，不依赖外部数据。真实题目可替换为空间温度、污染物浓度、水质扩散、土壤湿度或交通密度等空间-时间数据。

官方公开来源：

- Heat equation：<https://en.wikipedia.org/wiki/Heat_equation>
- Finite difference method：<https://en.wikipedia.org/wiki/Finite_difference_method>
- SciPy 线性代数文档：<https://docs.scipy.org/doc/scipy/reference/linalg.html>
