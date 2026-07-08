# PSO 数据与来源说明

## 内置演示数据

- `pso_benchmark.csv`：粒子群优化的 benchmark 配置文件，默认测试 Rastrigin 函数。

## 官方/经典来源

- Kennedy and Eberhart, 1995: *Particle Swarm Optimization*
- Clerc and Kennedy, 2002: *The Particle Swarm - Explosion, Stability, and Convergence...*

## 改良算法参考

- 动态惯性权重与混合策略综述：<https://arxiv.org/abs/1804.05319>
- 收缩因子理论：<https://ieeexplore.ieee.org/document/985692/>

## 使用要求

- 本示例默认是连续变量优化。
- 如果题目是离散优化，PSO 需要做离散化或改用其他方法。
