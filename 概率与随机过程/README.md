# 蒙特卡洛算法（2026-05-28）

```text
monte_carlo/
├── readme/
│   └── README.md
├── 代码/
│   ├── mc_baseline_pi_and_integral.py
│   ├── mc_improved_variance_reduction.py
│   ├── mc_quasi_monte_carlo.py
│   └── plot_mc_figures.py
├── figure/
└── dataset/
    ├── mc_demo_config.csv
    ├── mc_sources.md
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：蒙特卡洛（Monte Carlo）方法
- 核心思想：用随机抽样近似求解确定性难以直接处理的问题
- 核心公式：
  - 数值积分估计：`I ≈ (b-a) * (1/N) * Σ f(U_i)`
  - 概率估计：`P ≈ (1/N) * Σ 1{X_i ∈ A}`
  - 标准误差：`SE ∝ 1/sqrt(N)`
- 适用条件：
  - 维度较高、解析求解困难
  - 目标函数可模拟但难以求导
  - 需要估计概率、期望、风险或积分
- 常见失败场景：
  - 样本数太少，方差大，结果不稳定
  - 重要区域采样不足
  - 随机数质量差或随机种子不固定

### 蒙特卡洛的改良算法

| 改良方法                          | 如何改良                               | 相比原版的优点               | 相比原版的缺点           | 适用场景/数据集                     |
| --------------------------------- | -------------------------------------- | ---------------------------- | ------------------------ | ----------------------------------- |
| 重要性采样（Importance Sampling） | 改变抽样分布，让样本更多落在“重要区域” | 显著降方差，适合小概率事件   | 需要构造合适的提议分布   | 金融风险、罕见事件概率、尾部积分    |
| 分层抽样（Stratified Sampling）   | 把总体分层，每层都抽样                 | 样本更均衡，通常比纯随机更稳 | 分层规则需要设计         | 地域/人群分层、分段积分、结构化总体 |
| 拉丁超立方采样（LHS）             | 每一维都均匀覆盖区间                   | 低维-中维问题效果很好        | 维度过高时优势减弱       | 设计空间探索、参数不确定性分析      |
| 抗对称变量（Antithetic Variates） | 配对生成互补样本抵消波动               | 实现简单，能降方差           | 降方差幅度有限           | 单峰、平滑函数的积分和期望估计      |
| 控制变量法（Control Variates）    | 利用已知期望的相关变量修正估计         | 常能明显降方差               | 需要找到强相关控制变量   | 金融定价、仿真校准、积分估计        |
| 准蒙特卡洛（Quasi-Monte Carlo）   | 用低差异序列替代伪随机数               | 收敛通常更快、更均匀         | 理论与实现稍复杂         | 中低维数值积分、工程设计、模拟优化  |
| 马尔可夫链蒙特卡洛（MCMC）        | 构造马尔可夫链从目标分布采样           | 适合复杂后验分布和高维采样   | 需要收敛诊断，调参更复杂 | 贝叶斯推断、复杂后验、统计建模      |

**使用建议**

1. 如果只是“先做一个能跑通的随机估计”，先用基础 Monte Carlo。
2. 如果你发现结果波动大，优先考虑分层抽样、重要性采样或控制变量。
3. 如果是高维参数空间，优先考虑 LHS 或 Quasi-Monte Carlo。
4. 如果是复杂分布采样而不是简单积分，考虑 MCMC。

### B. 代码

- baseline 文件：`代码/mc_baseline_pi_and_integral.py`
- 改良文件：`代码/mc_improved_variance_reduction.py`
- 进一步改良文件：`代码/mc_quasi_monte_carlo.py`
- 可视化文件：`代码/plot_mc_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：估计误差、方差、收敛速度、重复实验稳定性
- 对比对象：
  - baseline Monte Carlo
  - variance reduction Monte Carlo
  - quasi-Monte Carlo
- 结论目标：证明改良版在相同样本量下更稳、误差更小

### D. 论文

- 方法段：

> 本文针对解析求解困难的数值估计问题采用蒙特卡洛方法进行求解。首先通过随机抽样构造样本序列，利用样本均值近似总体期望或积分值，从而实现概率与数值量的估计。考虑到基础蒙特卡洛方法存在收敛速度慢、方差较大的问题，进一步引入重要性采样、分层抽样与控制变量等降方差技术，并与准蒙特卡洛方法进行对比。实验结果表明，改良方法在相同样本量下具有更低估计误差和更稳定的重复实验表现，适用于国赛中的风险评估、小概率事件估计与复杂积分计算问题。

---

## 2. 蒙特卡洛理论补充

蒙特卡洛方法的本质是“用随机换解析”。当问题没有好看的闭式解、维度又很高时，随机模拟通常比精确求解更现实。

### 优点

- 适合高维问题
- 实现简单
- 易和其他方法结合
- 适合概率、期望、风险、积分估计

### 缺点

- 收敛速度一般是 `O(1/sqrt(N))`
- 结果存在随机波动
- 小概率事件估计容易不稳定
- 对随机数质量有依赖

### 适合场景

- 金融风险估计
- 排队系统与资源系统仿真
- 高维积分
- 不确定性分析
- 罕见事件概率估计

### 不太适合

- 有现成解析解的问题
- 对精度要求极高但样本预算有限的问题
- 构造高质量提议分布很困难的 MCMC 任务

---

## 3. 运行方式

```bash
python3 monte_carlo/代码/mc_baseline_pi_and_integral.py
python3 monte_carlo/代码/mc_improved_variance_reduction.py
python3 monte_carlo/代码/mc_quasi_monte_carlo.py
python3 monte_carlo/代码/plot_mc_figures.py
```

---

## 4. 参考来源

- NumPy random 官方文档：<https://numpy.org/doc/stable/reference/random/index.html>
- SciPy QMC 官方文档：<https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>
- Monte Carlo 方法综述：<https://en.wikipedia.org/wiki/Monte_Carlo_method>
- Importance Sampling 经典讲解：<https://arxiv.org/abs/1202.2242>
- MCMC 基础资料：<https://www.statlect.com/fundamentals-of-statistics/Markov-chain-Monte-Carlo>
