# 蒙特卡洛数据与来源说明

## 内置演示配置

- `mc_demo_config.csv`：包含圆周率估计和积分估计的示例配置。

## 官方/经典来源

- NumPy RNG 官方文档：<https://numpy.org/doc/stable/reference/random/index.html>
- SciPy QMC 官方文档：<https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>

## 改良算法参考

- 重要性采样：<https://arxiv.org/abs/1202.2242>
- MCMC 基础：<https://www.statlect.com/fundamentals-of-statistics/Markov-chain-Monte-Carlo>

## 使用要求

- 基础 Monte Carlo 依赖伪随机数。
- 改良方法通常需要根据问题结构设计抽样分布或分层方式。
- 如果是高维参数空间，优先考虑 LHS 或 Quasi-Monte Carlo。
