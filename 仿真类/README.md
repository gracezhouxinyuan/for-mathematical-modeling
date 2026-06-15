# 🎲 仿真类算法库 (Simulation Methods)

## 概述 Overview

仿真类算法用于**在不确定性条件下推演系统的演化**。主要处理随机性、概率分布、大规模系统行为等问题。

| 算法 | 应用场景 | 特点 | 复杂度 | 难度 |
|------|--------|------|--------|------|
| **蒙特卡洛** | 随机事件模拟 | 无偏估计，大量采样 | O(1/√N) | ⭐⭐ |
| **马尔可夫链** | 状态转移系统 | 无记忆性，稳定分布 | 中等 | ⭐⭐⭐ |
| **离散事件仿真(DES)** | 排队、生产 | 事件驱动，细粒度 | 高 | ⭐⭐⭐ |
| **Agent仿真** | 复杂系统 | 智能体交互，涌现性 | 很高 | ⭐⭐⭐⭐ |
| **马尔可夫链蒙特卡洛(MCMC)** | 贝叶斯推断 | 后验采样，灵活 | 高 | ⭐⭐⭐⭐ |

---

## 1. 蒙特卡洛方法 (Monte Carlo Simulation)

### 原理简介

通过**大量随机抽样**来近似求解问题。基于大数定律：样本均值趋向于期望。

**应用场景：**
- 估算圆周率π
- 积分计算
- 风险评估（VaR、CVaR）
- 投资组合分析

### 基础示例：估算π

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(n_samples=100000):
    """
    蒙特卡洛估算π
    
    在1×1正方形内，圆心为原点、半径为1的圆的面积为π/4
    落在圆内的点数占总点数的比例 ≈ π/4
    """
    # 在[0,1]²内随机采样
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # 距离原点的距离
    distances = np.sqrt(x**2 + y**2)
    
    # 落在单位圆内的点数
    inside_circle = np.sum(distances <= 1)
    
    # 估计π
    pi_estimate = 4 * inside_circle / n_samples
    
    return pi_estimate

# 运行模拟
pi_estimates = [monte_carlo_pi(1000) for _ in range(100)]
print(f"π的平均估计值: {np.mean(pi_estimates):.4f}")
print(f"真实值: {np.pi:.4f}")
print(f"估计标准差: {np.std(pi_estimates):.4f}")
```

### 积分计算

```python
def monte_carlo_integration(func, bounds, n_samples=100000):
    """
    蒙特卡洛方法计算定积分
    
    ∫∫ f(x,y) dx dy ≈ (体积) × (函数值均值)
    """
    # 多维采样
    n_dims = len(bounds)
    samples = np.random.uniform(0, 1, (n_samples, n_dims))
    
    # 归一化到实际范围
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = low + samples[:, i] * (high - low)
    
    # 计算函数值
    func_values = np.array([func(*sample) for sample in samples])
    
    # 体积
    volume = np.prod([high - low for low, high in bounds])
    
    # 积分估计
    integral = volume * np.mean(func_values)
    
    return integral, np.std(func_values) * volume / np.sqrt(n_samples)

# 示例：计算 ∫∫ (x² + y²) dx dy，x,y ∈ [0,1]
func = lambda x, y: x**2 + y**2
bounds = [(0, 1), (0, 1)]
integral, error = monte_carlo_integration(func, bounds)
print(f"积分值: {integral:.4f} ± {error:.4f}")
```

### 金融风险评估

```python
def monte_carlo_var(returns, confidence=0.95, n_simulations=10000):
    """
    蒙特卡洛模拟计算风险价值(VaR)和条件风险价值(CVaR)
    """
    # 拟合历史收益分布
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # 蒙特卡洛模拟
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
    # VaR: 在给定置信度下的损失下限
    var = np.percentile(simulated_returns, (1 - confidence) * 100)
    
    # CVaR: VaR之外的平均损失
    cvar = np.mean(simulated_returns[simulated_returns <= var])
    
    # 绘制分布
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_returns, bins=50, alpha=0.7, edgecolor='k')
    plt.axvline(var, color='red', linestyle='--', linewidth=2, label=f'VaR({confidence*100:.0f}%)={var:.4f}')
    plt.axvline(cvar, color='orange', linestyle='--', linewidth=2, label=f'CVaR={cvar:.4f}')
    plt.xlabel('收益率')
    plt.ylabel('频数')
    plt.legend()
    plt.title('蒙特卡洛模拟收益分布')
    plt.show()
    
    return var, cvar

# 示例
historical_returns = np.random.normal(0.05, 0.15, 252)  # 一年的每日收益
var, cvar = monte_carlo_var(historical_returns)
print(f"VaR: {var:.4f}, CVaR: {cvar:.4f}")
```

---

## 2. 马尔可夫链 (Markov Chain)

### 原理简介

无记忆性随机过程：**下一状态仅取决于当前状态，与历史无关**。

$$P(X_{n+1} = j | X_n = i, X_{n-1}, ...) = P(X_{n+1} = j | X_n = i) = P_{ij}$$

**核心概念：**
- **转移矩阵 P**: $P_{ij}$ 是从状态i到状态j的转移概率
- **平稳分布 π**: $\pi P = \pi$，长期分布
- **首达时间**：第一次达到目标状态的时间

### Python实现

```python
import numpy as np
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, transition_matrix, initial_state=None):
        """
        Args:
            transition_matrix: n×n转移矩阵
            initial_state: 初始状态分布 (默认均匀)
        """
        self.P = np.array(transition_matrix)
        self.n_states = self.P.shape[0]
        
        if initial_state is None:
            self.state_dist = np.ones(self.n_states) / self.n_states
        else:
            self.state_dist = np.array(initial_state)
    
    def simulate(self, n_steps):
        """模拟马尔可夫链演化"""
        states = np.zeros(n_steps, dtype=int)
        # 初始状态采样
        states[0] = np.random.choice(self.n_states, p=self.state_dist)
        
        for t in range(n_steps - 1):
            # 从当前状态转移到下一状态
            states[t+1] = np.random.choice(self.n_states, p=self.P[states[t]])
        
        return states
    
    def stationary_distribution(self):
        """计算平稳分布（特征向量）"""
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        
        # 找到特征值为1的特征向量
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)
        
        return stationary
    
    def n_step_transition(self, n):
        """计算n步转移矩阵"""
        return np.linalg.matrix_power(self.P, n)
    
    def expected_first_passage_time(self, target_state):
        """计算到达目标状态的期望时间"""
        n_states = self.n_states
        Q = self.P[:n_states-1, :n_states-1]  # 去掉目标状态的子矩阵
        I = np.eye(n_states - 1)
        
        # (I - Q)^(-1) 的每行和
        N = np.linalg.inv(I - Q)
        expected_times = np.sum(N, axis=1)
        
        return expected_times

# 示例：天气模型
# 状态：0=晴天, 1=多云, 2=下雨
transition_matrix = [
    [0.8, 0.15, 0.05],   # 晴天→晴天0.8, →多云0.15, →下雨0.05
    [0.3, 0.5, 0.2],     # 多云
    [0.2, 0.3, 0.5]      # 下雨
]

mc = MarkovChain(transition_matrix)

# 模拟100天的天气
weather_states = mc.simulate(100)
state_names = ['晴天', '多云', '下雨']
print("前20天天气:", [state_names[s] for s in weather_states[:20]])

# 平稳分布
stationary = mc.stationary_distribution()
print(f"平稳分布: 晴天{stationary[0]:.3f}, 多云{stationary[1]:.3f}, 下雨{stationary[2]:.3f}")

# 10步后的分布
p10 = mc.n_step_transition(10)
print(f"10步转移矩阵:\n{p10}")
```

### 应用：排队论

```python
class QueueingSystem:
    """M/M/1排队系统"""
    
    def __init__(self, lambda_, mu):
        """
        Args:
            lambda_: 到达率
            mu: 服务率
        """
        self.lambda_ = lambda_
        self.mu = mu
        self.rho = lambda_ / mu  # 利用率
    
    def steady_state_probs(self, max_n=20):
        """计算稳定状态下系统中有n个顾客的概率"""
        if self.rho >= 1:
            print("系统不稳定（λ >= μ）")
            return None
        
        probs = [(1 - self.rho) * self.rho ** n for n in range(max_n)]
        return np.array(probs)
    
    def performance_metrics(self):
        """计算性能指标"""
        rho = self.rho
        L = rho / (1 - rho)  # 系统平均顾客数
        W = 1 / (self.mu - self.lambda_)  # 平均停留时间
        Lq = rho**2 / (1 - rho)  # 队列平均顾客数
        Wq = rho / (self.mu - self.lambda_)  # 平均等待时间
        
        return {
            'L': L,  # 系统中平均顾客数
            'W': W,  # 平均停留时间
            'Lq': Lq,  # 队列中平均顾客数
            'Wq': Wq  # 平均等待时间
        }

# 示例
queue = QueueingSystem(lambda_=4, mu=5)
metrics = queue.performance_metrics()
print(f"系统平均顾客数(L): {metrics['L']:.2f}")
print(f"平均停留时间(W): {metrics['W']:.2f}小时")
```

---

## 3. 离散事件仿真 (Discrete Event Simulation, DES)

### 概念

通过**事件驱动**来模拟系统中的离散变化。

```python
from heapq import heappush, heappop
import numpy as np

class DiscreteEventSimulation:
    """离散事件仿真框架"""
    
    def __init__(self, seed=None):
        self.event_queue = []  # 优先队列：(time, event_id, event_type, data)
        self.current_time = 0
        self.event_counter = 0
        if seed is not None:
            np.random.seed(seed)
    
    def schedule_event(self, event_type, delay, data=None):
        """调度事件"""
        event_time = self.current_time + delay
        event_id = self.event_counter
        self.event_counter += 1
        heappush(self.event_queue, (event_time, event_id, event_type, data))
    
    def run(self, stop_time, event_handler):
        """运行仿真"""
        while self.event_queue and self.current_time < stop_time:
            event_time, event_id, event_type, data = heappop(self.event_queue)
            self.current_time = event_time
            
            event_handler(event_type, data, self)
    
    def get_current_time(self):
        return self.current_time

# 示例：银行排队
class BankQueue:
    def __init__(self):
        self.queue_length = 0
        self.total_wait_time = 0
        self.customers_served = 0
    
    def event_handler(self, event_type, data, sim):
        if event_type == 'arrival':
            self.queue_length += 1
            print(f"时间 {sim.get_current_time():.2f}: 顾客到达，队列长度={self.queue_length}")
            
            # 调度下一个到达事件（指数分布）
            inter_arrival = np.random.exponential(scale=1.0)
            sim.schedule_event('arrival', inter_arrival)
            
            # 如果队列只有这一个顾客，开始服务
            if self.queue_length == 1:
                service_time = np.random.exponential(scale=0.8)
                sim.schedule_event('finish', service_time)
        
        elif event_type == 'finish':
            self.queue_length -= 1
            self.customers_served += 1
            print(f"时间 {sim.get_current_time():.2f}: 顾客离开，队列长度={self.queue_length}")
            
            # 如果还有客户，继续服务
            if self.queue_length > 0:
                service_time = np.random.exponential(scale=0.8)
                sim.schedule_event('finish', service_time)

# 运行仿真
bank = BankQueue()
sim = DiscreteEventSimulation(seed=42)
sim.schedule_event('arrival', 0)  # 第一个顾客立即到达
sim.run(stop_time=100, event_handler=bank.event_handler)
print(f"总共服务顾客数: {bank.customers_served}")
```

---

## 4. Agent仿真

Agent仿真用于建模**多个自主智能体的交互**，产生涌现现象。

```python
class Agent:
    """基础智能体类"""
    
    def __init__(self, agent_id, x, y):
        self.id = agent_id
        self.x = x
        self.y = y
        self.velocity = np.random.randn(2)
    
    def step(self, world):
        """一步行动"""
        # 更新位置
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # 边界处理（环形）
        self.x = self.x % world.width
        self.y = self.y % world.height
        
        # 随机偏向（Brownian运动）
        self.velocity += np.random.randn(2) * 0.1

class World:
    """环境世界"""
    
    def __init__(self, width=100, height=100, n_agents=50):
        self.width = width
        self.height = height
        self.agents = [
            Agent(i, np.random.uniform(0, width), np.random.uniform(0, height))
            for i in range(n_agents)
        ]
    
    def step(self):
        """一个时间步"""
        for agent in self.agents:
            agent.step(self)
    
    def simulate(self, n_steps=1000):
        """运行仿真"""
        positions_history = []
        
        for step in range(n_steps):
            self.step()
            positions = np.array([[a.x, a.y] for a in self.agents])
            positions_history.append(positions)
        
        return np.array(positions_history)

# 运行agent仿真
world = World(width=100, height=100, n_agents=30)
history = world.simulate(n_steps=500)

# 可视化最后一帧
plt.figure(figsize=(8, 8))
plt.scatter(history[-1, :, 0], history[-1, :, 1], s=50, alpha=0.6)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Agent仿真最终分布')
plt.show()
```

---

## 5. MCMC (Markov Chain Monte Carlo)

用于从复杂分布中采样，常用于贝叶斯推断。

```python
def metropolis_hastings(target_pdf, proposal_std=1.0, n_iterations=10000, initial_x=0):
    """
    Metropolis-Hastings算法
    
    Args:
        target_pdf: 目标概率密度函数
        proposal_std: 提议分布的标准差
        n_iterations: 迭代次数
        initial_x: 初始值
    
    Returns:
        samples: 采样结果
    """
    samples = np.zeros(n_iterations)
    current_x = initial_x
    n_accept = 0
    
    for i in range(n_iterations):
        # 从提议分布采样
        proposed_x = current_x + np.random.normal(0, proposal_std)
        
        # 计算接受概率
        acceptance_ratio = target_pdf(proposed_x) / target_pdf(current_x)
        
        # Metropolis步骤
        if np.random.rand() < acceptance_ratio:
            current_x = proposed_x
            n_accept += 1
        
        samples[i] = current_x
    
    acceptance_rate = n_accept / n_iterations
    print(f"接受率: {acceptance_rate:.3f}")
    
    return samples

# 示例：从混合高斯分布采样
def target_pdf(x):
    return 0.3 * np.exp(-0.5*(x-2)**2) + 0.7 * np.exp(-0.5*(x+2)**2)

samples = metropolis_hastings(target_pdf, proposal_std=0.5)

# 绘制结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(samples[:1000])
plt.xlabel('迭代次数')
plt.ylabel('采样值')
plt.title('MCMC采样序列（前1000步）')

plt.subplot(1, 2, 2)
plt.hist(samples[1000:], bins=50, density=True, alpha=0.7, label='采样分布')
x = np.linspace(-6, 6, 1000)
plt.plot(x, target_pdf(x), 'r-', linewidth=2, label='目标分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.legend()
plt.title('收敛到目标分布')

plt.tight_layout()
plt.show()
```

---

## 论文输出形式

1. **仿真过程展示** → 动画或关键时刻截图
2. **分布收敛** → 仿真步数 vs 关键指标
3. **敏感性分析** → 参数变化 vs 结果变化的曲线
4. **置信区间** → 结果±标准差的表格
5. **对比分析** → 理论值 vs 仿真值

---

## 调试技巧

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 结果波动大 | 采样不足 | 增加仿真次数 |
| 收敛慢 | 初值不合理 | 改进初值策略 |
| 内存溢出 | 记录数据过多 | 只记录关键统计量 |
| 结果与理论差距大 | 模型假设不符 | 检查模型逻辑 |

---

## 延伸学习

- **变差递减(Variance Reduction)**：重要性采样、分层采样
- **准蒙特卡洛(QMC)**：低差异序列（Sobol、Halton）
- **粒子滤波**：非线性滤波
- **模拟退火中的MCMC**：参数优化
