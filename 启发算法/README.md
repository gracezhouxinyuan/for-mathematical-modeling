# PSO（2026-05-25）

```text
pso/
├── readme/
│   └── README.md
├── 代码/
│   ├── pso_baseline.py
│   ├── pso_improved_inertia_topology.py
│   └── plot_pso_figures.py
├── figure/
└── dataset/
    ├── pso_benchmark.csv
    ├── pso_sources.md
    └── dataset_sources.md
```

---

## 1. 今日任务

### A. 理论

- 学习算法：PSO（粒子群优化）
- 核心思想：粒子在搜索空间中跟随个体最优和全局最优移动，通过速度和位置更新找到最优解
- 核心公式：
  - `v(t+1) = w*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)`
  - `x(t+1) = x(t) + v(t+1)`
- 适用条件：
  - 连续参数优化
  - 黑箱优化、无梯度问题
  - 需要较快的工程寻优结果
- 常见失败场景：
  - 过快收敛到局部最优
  - 高维问题多样性不足
  - 参数设置不合理导致振荡或发散

### PSO 的改良算法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
|---|---|---|---|---|
| 惯性权重动态 PSO | 让 `w` 随迭代逐步下降或非线性变化 | 前期探索强，后期收敛稳 | 需要设计权重衰减策略 | 连续函数优化、参数寻优、神经网络权重调节 |
| Clerc 收缩因子 PSO | 用收缩因子稳定速度更新，抑制爆炸 | 收敛性更好，理论更稳定 | 需要满足参数约束条件 | 高维连续优化、需要稳定收敛的场景 |
| 局部拓扑 PSO（lbest / ring） | 不是只跟随全局最优，而是跟随邻域最优 | 多样性更好，减少早熟 | 收敛速度可能慢一些 | 多峰函数、容易陷入局部最优的问题 |
| Quantum-behaved PSO（QPSO） | 用量子行为替代经典速度更新 | 搜索更跳跃，跳出局部能力强 | 理解和调参稍复杂 | 多峰、复杂地形的连续优化 |
| 混合 PSO-局部搜索 | PSO 找全局区域，局部搜索做精修 | 精度更高，论文也更好写 | 算法链条更长 | 连续参数估计、拟合、工程优化 |

**使用建议**

1. 只要是连续变量优化，PSO 都值得先试。
2. 如果总是早熟，优先用动态惯性权重或局部拓扑。
3. 如果要写“理论更稳”的改良，优先 Clerc 收缩因子。
4. 如果目标函数多峰且复杂，考虑 QPSO 或混合局部搜索。

### B. 代码

- baseline 文件：`代码/pso_baseline.py`
- 改良文件：`代码/pso_improved_inertia_topology.py`
- 可视化文件：`代码/plot_pso_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：最优值、收敛速度、稳定性、最终位置误差
- 对比对象：
  - baseline gbest PSO
  - improved inertia-topology PSO
- 结论目标：证明改良版更稳、更不容易振荡，也更适合多峰函数

### D. 论文

- 方法段：

> 本文针对连续参数优化问题构建粒子群优化（PSO）模型。首先初始化粒子位置与速度，并通过个体最优和全局最优引导搜索方向。考虑到标准 PSO 容易在多峰函数上过早收敛，进一步引入动态惯性权重与局部拓扑结构，以增强前期探索能力并维持群体多样性。实验结果表明，改良后的 PSO 在相同迭代预算下具有更好的收敛稳定性和更低的最优值误差，适合国赛中的连续参数寻优、拟合与工程优化任务。

---

## 2. PSO 理论补充

PSO 的优势是实现非常简单，参数也不多，所以它在国赛里特别适合做“连续优化的第一选择”。但它的最大问题是容易跟着全局最优一起陷进去，导致群体越来越像、越来越早熟。

### 适合场景

- 连续函数优化
- 神经网络参数搜索
- PID 参数整定
- 拟合与校准
- 工程设计参数优化

### 不太适合

- 强离散问题
- 组合结构特别明显的问题
- 高噪声且评估代价很高的问题

---

## 3. 运行方式

```bash
python3 pso/代码/pso_baseline.py
python3 pso/代码/pso_improved_inertia_topology.py
python3 pso/代码/plot_pso_figures.py
```

---

## 4. 参考来源

- Kennedy and Eberhart, 1995: Particle Swarm Optimization
- Clerc and Kennedy, 2002: The Particle Swarm - Explosion, Stability, and Convergence
- PSO 综述：<https://arxiv.org/abs/1804.05319>
- IEEE Xplore 经典论文页：<https://ieeexplore.ieee.org/document/985692/>
