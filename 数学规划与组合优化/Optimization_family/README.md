# LP / IP / MILP / QP / NLP（2026-06-01）

```text
optimization_family/
├── readme/
│   └── README.md
├── 代码/
│   ├── lp_baseline.py
│   ├── ip_milp_knapsack.py
│   ├── qp_portfolio.py
│   ├── nlp_rosenbrock.py
│   └── plot_optimization_figures.py
├── figure/
└── dataset/
    ├── lp_data.csv
    ├── milp_knapsack.csv
    ├── qp_data.csv
    ├── nlp_data.csv
    ├── optimization_sources.md
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：
  - LP：线性规划
  - IP：整数规划
  - MILP：混合整数线性规划
  - QP：二次规划
  - NLP：非线性规划
- 核心思想：
  - LP：线性目标 + 线性约束
  - IP/MILP：在 LP 上加入整数变量约束
  - QP：目标函数含二次项
  - NLP：目标或约束含非线性项
- 核心公式：
  - LP: `min c^T x`
  - QP: `min 1/2 x^T P x + q^T x`
  - NLP: `min f(x)` subject to `g(x) <= 0, h(x)=0`
- 适用条件：
  - LP：资源分配、成本最小、运输计划
  - IP/MILP：离散决策、选址、排产、背包
  - QP：投资组合、最小方差、二次成本
  - NLP：复杂物理/工程约束、非线性拟合
- 常见失败场景：
  - 模型写成线性却忘了整数约束
  - 变量尺度差异太大导致数值不稳定
  - 非凸 NLP 局部最优严重

### 各类模型的改良算法

| 模型                    | 如何改良                                                     | 相比原版的优点              | 相比原版的缺点               | 适用场景/数据集                  |
| ----------------------- | ------------------------------------------------------------ | --------------------------- | ---------------------------- | -------------------------------- |
| LP                      | 对偶单纯形、内点法、预处理、分解法（Benders/Dantzig-Wolfe）  | 大规模稀疏问题更稳更快      | 建模需保持线性结构           | 运输、生产、配比、资源分配       |
| IP / MILP               | Branch and Bound、Branch and Cut、Cutting Planes、启发式 warm start | 能处理离散决策，求解更完整  | NP-hard，规模大时计算慢      | 选址、排产、背包、0-1 决策       |
| QP                      | Active Set、Interior Point、Convex Relaxation、MIQP 扩展     | 二次目标可自然表达风险/方差 | 非凸 QP 难度高               | 投资组合、二次成本、控制问题     |
| NLP                     | SQP、Augmented Lagrangian、Penalty Method、Trust Region      | 更适合非线性约束与曲面目标  | 可能陷入局部最优，对初值敏感 | 参数拟合、工程优化、复杂约束系统 |
| LP/MILP/QP/NLP 统一建模 | 规模化求解器（HiGHS、Gurobi、MOSEK、CVXPY）                  | 代码更简洁，工程可落地      | 依赖求解器环境               | 国赛论文与工程原型               |

**使用建议**

1. 能线性就尽量线性，优先 LP。
2. 一旦有整数/0-1 决策，考虑 IP/MILP。
3. 若目标是风险、方差、二次惩罚，考虑 QP。
4. 若模型非线性强，先判断是否可近似成 LP/MILP/QP，再上 NLP。

### B. 代码

- baseline 文件：`代码/lp_baseline.py`
- 改良文件：`代码/ip_milp_knapsack.py`
- 进一步改良文件：`代码/qp_portfolio.py`
- 非线性示例：`代码/nlp_rosenbrock.py`
- 可视化文件：`代码/plot_optimization_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：目标值、可行性、求解时间、稳定性
- 对比对象：
  - LP baseline
  - MILP integer extension
  - QP portfolio model
  - NLP Rosenbrock demo
- 结论目标：展示不同规划模型的表达能力和求解特征差异

### D. 论文

- 方法段：

> 本文针对不同类型的优化问题构建 LP、IP/MILP、QP 与 NLP 的统一建模框架。对于线性资源分配问题，采用线性规划进行求解；对于包含离散决策的场景，引入整数约束构建混合整数规划模型；对于带有风险或二次成本项的问题，进一步采用二次规划刻画目标函数；对于具有显著非线性特征的问题，则采用非线性规划并结合适当求解器进行数值优化。通过统一的建模与求解流程，本文展示了不同规划模型在表达能力、求解效率和适用场景上的差异，并为国赛中的资源分配、选址、组合优化和工程优化问题提供了可复用框架。

---

## 2. 理论补充

### LP

线性规划的核心是“目标线性、约束线性”。它特别适合建模资源、成本、产能、运输、配料等问题。

### IP / MILP

当决策变量必须是整数或 0-1 时，就需要 IP/MILP。它比 LP 更真实，但求解难度也明显更高。

### QP

当目标中有平方项，比如投资风险、最小方差、二次成本时，就适合 QP。很多国赛的“风险-收益平衡”都可以写成 QP 或 MIQP。

### NLP

当目标或约束是非线性的，比如物理模型、拟合模型、复杂约束系统，就需要 NLP。它最强，但对初值和局部最优最敏感。

---

## 3. 运行方式

```bash
python3 optimization_family/代码/lp_baseline.py
python3 optimization_family/代码/ip_milp_knapsack.py
python3 optimization_family/代码/qp_portfolio.py
python3 optimization_family/代码/nlp_rosenbrock.py
python3 optimization_family/代码/plot_optimization_figures.py
```

---

## 4. 参考来源

- SciPy `linprog` 官方文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>
- SciPy `minimize` 官方文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>
- SciPy 优化总览：<https://docs.scipy.org/doc/scipy/tutorial/optimize.html>
- CVXPY 求解器文档：<https://www.cvxpy.org/tutorial/solvers/index.html>
- CVXPY 约束与混合整数支持：<https://www.cvxpy.org/tutorial/constraints/index.html>
- HiGHS 优化求解器：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>
