# 优化模型数据与来源说明

## 内置演示数据

- `lp_data.csv`：线性规划演示数据
- `milp_knapsack.csv`：整数/混合整数背包示例
- `qp_data.csv`：二次规划/投资组合示例
- `nlp_data.csv`：非线性优化示例配置

## 官方文档

- SciPy `linprog`：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>
- SciPy `minimize`：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>
- CVXPY 求解器：<https://www.cvxpy.org/tutorial/solvers/index.html>

## 改良算法参考

- HiGHS：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>
- CVXPY mixed-integer：<https://www.cvxpy.org/tutorial/constraints/index.html>

## 使用要求

- LP 需要线性结构。
- IP/MILP 需要整数约束。
- QP 需要二次目标或二次约束。
- NLP 适合非线性目标/约束，但要注意局部最优。
