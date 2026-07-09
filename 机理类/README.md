# 🔬 机理类算法库 (Mechanism & Dynamics Models)

## 概述 Overview

机理类算法用于**建立微分方程模型描述动态过程**，如传播、流行病、化学反应等。需要深入理解系统的物理或生物学过程。

| 算法           | 应用场景       | 特点                | 难度 |
| -------------- | -------------- | ------------------- | ---- |
| **ODE**        | 单变量动态过程 | 常微分方程,初值问题 | ⭐⭐   |
| **SIR/SEIR**   | 传染病传播     | 分室模型,参数清晰   | ⭐⭐⭐  |
| **PDE**        | 空间-时间演化  | 偏微分方程,复杂求解 | ⭐⭐⭐⭐ |
| **参数估计**   | 模型拟合       | 反演问题,优化求解   | ⭐⭐⭐  |
| **敏感性分析** | 参数影响评估   | 量化参数效应        | ⭐⭐   |
| **稳定性分析** | 平衡态研究     | 线性化,特征值       | ⭐⭐⭐  |

---

## 1. 常微分方程 (ODE, Ordinary Differential Equations)

### 基本形式

$$\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$$

### 数值求解方法

#### Euler方法（最简单）

```python
def euler_method(f, y0, t_span, h=0.01):
    """
    Euler前向差分法
    
    Args:
        f: 微分方程 dy/dt = f(t, y)
        y0: 初值
        t_span: (t_start, t_end)
        h: 步长
    
    Returns:
        t: 时间数组
        y: 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i])
    
    return t, y

# 示例：dy/dt = -0.5*y，初值y(0)=1
f = lambda t, y: -0.5 * y
t, y = euler_method(f, y0=1, t_span=(0, 10))
```

#### Runge-Kutta方法（4阶，更精确）

```python
def rk4_method(f, y0, t_span, h=0.01):
    """RK4方法"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + k1*h/2)
        k3 = f(t[i] + h/2, y[i] + k2*h/2)
        k4 = f(t[i] + h, y[i] + k3*h)
        
        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y
```

#### 使用scipy求解（推荐）

```python
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# 定义微分方程
def model(y, t):
    dydt = -0.5 * y
    return dydt

# 初值与时间序列
y0 = 1
t = np.linspace(0, 10, 100)

# 求解
y = odeint(model, y0, t)

# 绘图
plt.plot(t, y)
plt.xlabel('时间')
plt.ylabel('y(t)')
plt.show()
```

---

## 2. SIR模型（传染病）

### 原理

将人口分为三类：

- **S (Susceptible)**: 易感者
- **I (Infected)**: 感染者
- **R (Recovered)**: 康复者（获得免疫）

### 微分方程

$$\frac{dS}{dt} = -\beta \frac{SI}{N}$$
$$\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

其中：

- **β**: 感染率（日接触数 × 单次传播概率）
- **γ**: 恢复率（1/γ是平均患病时间）
- **R₀ = β/γ**: 基本再生数

### Python实现

```python
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class SIRModel:
    def __init__(self, beta, gamma, N):
        """
        Args:
            beta: 感染率
            gamma: 恢复率
            N: 总人口数
        """
        self.beta = beta
        self.gamma = gamma
        self.N = N
    
    def model(self, y, t):
        S, I, R = y
        dS_dt = -self.beta * S * I / self.N
        dI_dt = self.beta * S * I / self.N - self.gamma * I
        dR_dt = self.gamma * I
        return [dS_dt, dI_dt, dR_dt]
    
    def solve(self, S0, I0, R0, t_span):
        """
        求解SIR模型
        
        Args:
            S0, I0, R0: 初始易感、感染、康复人数
            t_span: (start, end) 时间范围
        
        Returns:
            t, S, I, R: 时间序列及各类人数
        """
        y0 = [S0, I0, R0]
        t = np.linspace(t_span[0], t_span[1], 1000)
        result = odeint(self.model, y0, t)
        
        return t, result[:, 0], result[:, 1], result[:, 2]
    
    def plot(self, t, S, I, R):
        """绘制结果"""
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label='Susceptible (S)')
        plt.plot(t, I, label='Infected (I)')
        plt.plot(t, R, label='Recovered (R)')
        plt.xlabel('时间 (天)')
        plt.ylabel('人数')
        plt.legend()
        plt.grid(True)
        plt.show()

# 使用示例
model = SIRModel(beta=0.5, gamma=0.1, N=10000)
t, S, I, R = model.solve(S0=9900, I0=100, R0=0, t_span=(0, 160))
model.plot(t, S, I, R)
```

---

## 3. SEIR模型（潜伏期）

### 原理

在SIR基础上加入**潜伏期**，引入E(Exposed)类：

- **S**: 易感者
- **E**: 潜伏者（已感染但未发病）
- **I**: 感染者（发病）
- **R**: 康复者

### 微分方程

$$\frac{dS}{dt} = -\beta \frac{SI}{N}$$
$$\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E$$
$$\frac{dI}{dt} = \sigma E - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

其中σ是潜伏期转化率（1/σ是平均潜伏期）。

### Python实现

```python
class SEIRModel:
    def __init__(self, beta, sigma, gamma, N):
        """
        Args:
            beta: 感染率
            sigma: 潜伏期转化率
            gamma: 恢复率
            N: 总人口
        """
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
    
    def model(self, y, t):
        S, E, I, R = y
        dS_dt = -self.beta * S * I / self.N
        dE_dt = self.beta * S * I / self.N - self.sigma * E
        dI_dt = self.sigma * E - self.gamma * I
        dR_dt = self.gamma * I
        return [dS_dt, dE_dt, dI_dt, dR_dt]
    
    def solve(self, S0, E0, I0, R0, t_span):
        y0 = [S0, E0, I0, R0]
        t = np.linspace(t_span[0], t_span[1], 1000)
        result = odeint(self.model, y0, t)
        return t, result[:, 0], result[:, 1], result[:, 2], result[:, 3]
```

---

## 4. 参数估计（模型拟合）

### 问题

给定实际数据，反推微分方程的参数。

### 方法：最小二乘法

```python
from scipy.optimize import least_squares
from scipy.integrate import odeint

def estimate_parameters(data_t, data_y, initial_params):
    """
    参数估计 - 最小二乘法
    
    Args:
        data_t: 实验时间点
        data_y: 实验测量值
        initial_params: 参数初值
    
    Returns:
        estimated_params: 估计的参数
    """
    
    def residual(params):
        """残差函数"""
        def model(y, t):
            k = params[0]  # 模型参数
            return -k * y
        
        y_pred = odeint(model, data_y[0], data_t)[:, 0]
        return y_pred - data_y
    
    result = least_squares(residual, initial_params)
    return result.x

# 示例：已知数据，估计k值
data_t = np.array([0, 1, 2, 3, 4, 5])
data_y = np.array([1.0, 0.82, 0.67, 0.55, 0.45, 0.37])  # 指数衰减
k_est = estimate_parameters(data_t, data_y, initial_params=[0.1])
print(f"估计的k值: {k_est[0]:.4f}")
```

### 高级方法：遗传算法 + 微分方程

```python
def parameter_estimation_ga(data_t, data_y, bounds):
    """使用遗传算法估计参数"""
    from scipy.optimize import differential_evolution
    
    def objective(params):
        """目标函数：拟合误差平方和"""
        def model(y, t):
            return -params[0] * y
        
        y_pred = odeint(model, data_y[0], data_t)[:, 0]
        return np.sum((y_pred - data_y)**2)
    
    result = differential_evolution(objective, bounds)
    return result.x
```

---

## 5. 敏感性分析

### 目的

评估参数变化对模型输出的影响。

### 本地敏感性（一阶导数）

```python
def sensitivity_analysis_local(model_func, params, dy_dt_func, t):
    """
    局部敏感性分析
    
    Args:
        model_func: ODE模型函数
        params: 参数字典
        dy_dt_func: ODE右端函数
        t: 时间序列
    
    Returns:
        sensitivity: 参数的敏感度系数
    """
    
    # 设置基准模型
    y_base = odeint(dy_dt_func, y0, t)
    
    sensitivity = {}
    for param_name in params:
        # 参数摄动
        delta = 0.01 * params[param_name]
        params[param_name] += delta
        
        y_perturb = odeint(dy_dt_func, y0, t)
        
        # 敏感度系数
        S = (y_perturb - y_base) / delta / y_base
        sensitivity[param_name] = np.max(np.abs(S))
        
        # 恢复参数
        params[param_name] -= delta
    
    return sensitivity
```

### 全局敏感性（Sobol指数）

```python
def sobol_sensitivity(model_func, param_ranges, n_samples=1000):
    """
    Sobol全局敏感性分析（简化版）
    """
    import numpy as np
    from scipy.stats.qmc import Sobol
    
    # 使用Sobol序列采样
    sampler = Sobol(d=len(param_ranges))
    samples = sampler.random(n_samples)
    
    # 归一化到参数范围
    for i, (low, high) in enumerate(param_ranges):
        samples[:, i] = low + samples[:, i] * (high - low)
    
    # 评估模型
    outputs = np.array([model_func(sample) for sample in samples])
    
    # 计算方差贡献
    var_total = np.var(outputs)
    sobol_indices = {}
    
    for i in range(len(param_ranges)):
        # 一阶指数 (简化计算)
        grouped = outputs.reshape(-1, 10)  # 分组
        var_param = np.var(grouped.mean(axis=1))
        sobol_indices[i] = var_param / var_total
    
    return sobol_indices
```

---

## 6. 稳定性分析

### 平衡点与线性化

对于系统 $\frac{dy}{dt} = f(y)$，如果 $f(y^*) = 0$，则 $y^*$ 是平衡点。

线性化：$\frac{dz}{dt} = J(y^*) z$，其中 $J$ 是Jacobian矩阵。

### 稳定性判定

```python
def stability_analysis(f, y_equilibrium, delta=1e-5):
    """
    线性稳定性分析
    
    Args:
        f: ODE右端函数 (向量形式)
        y_equilibrium: 平衡点
        delta: 数值微分步长
    
    Returns:
        eigenvalues: Jacobian矩阵的特征值
        is_stable: 是否稳定
    """
    import numpy as np
    
    n = len(y_equilibrium)
    J = np.zeros((n, n))
    
    # 数值计算Jacobian
    for i in range(n):
        y_plus = y_equilibrium.copy()
        y_plus[i] += delta
        
        y_minus = y_equilibrium.copy()
        y_minus[i] -= delta
        
        J[:, i] = (f(y_plus) - f(y_minus)) / (2 * delta)
    
    # 特征值
    eigenvalues = np.linalg.eigvals(J)
    
    # 稳定性判定
    is_stable = np.all(eigenvalues.real < 0)
    
    return eigenvalues, is_stable

# 示例：SIR模型的平衡点分析
def sir_f(y):
    S, I, R = y
    beta, gamma, N = 0.5, 0.1, 10000
    return np.array([
        -beta * S * I / N,
        beta * S * I / N - gamma * I,
        gamma * I
    ])

# 无病平衡点
y_eq = np.array([10000, 0, 0])
eigenvalues, is_stable = stability_analysis(sir_f, y_eq)
print(f"特征值: {eigenvalues}")
print(f"稳定性: {'稳定' if is_stable else '不稳定'}")
```

---

## 7. 数值方法对比

| 方法         | 精度   | 计算效率 | 适用场景   |
| ------------ | ------ | -------- | ---------- |
| Euler        | 一阶   | 快       | 快速原型   |
| RK2          | 二阶   | 中       | 中等精度   |
| RK4          | 四阶   | 中       | 标准选择   |
| RK45(自适应) | 四五阶 | 慢       | 高精度需求 |

---

## 论文常用输出形式

### 1. 动态曲线图

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：各类人数随时间变化
axes[0].plot(t, S, label='S', linewidth=2)
axes[0].plot(t, I, label='I', linewidth=2)
axes[0].plot(t, R, label='R', linewidth=2)
axes[0].set_xlabel('时间(天)')
axes[0].set_ylabel('人数')
axes[0].legend()
axes[0].grid(True)

# 右图：相图 (I vs S)
axes[1].plot(S, I, linewidth=2)
axes[1].set_xlabel('易感者数(S)')
axes[1].set_ylabel('感染者数(I)')
axes[1].set_title('SIR相图')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### 2. 参数影响对比表

| 参数 | 值1  | 值2  | 值3  | 关键指标变化 |
| ---- | ---- | ---- | ---- | ------------ |
| β    | 0.3  | 0.5  | 0.7  | 峰值增加     |
| γ    | 0.05 | 0.1  | 0.15 | 峰值降低     |

### 3. 模型拟合图

```python
# 实验数据 vs 模型预测
plt.figure()
plt.plot(data_t, data_y, 'ro', label='实验数据', markersize=8)
plt.plot(t, y_pred, 'b-', label='模型预测', linewidth=2)
plt.xlabel('时间')
plt.ylabel('浓度')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 常见问题与调试

| 问题             | 原因             | 解决方案               |
| ---------------- | ---------------- | ---------------------- |
| 数值不稳定、振荡 | 步长过大         | 减小h或使用自适应步长  |
| 计算结果不收敛   | 参数设置不合理   | 检查初值、参数范围     |
| 拟合误差大       | 模型假设不符     | 改进模型结构或参数     |
| 特征值计算错误   | Jacobian矩阵精度 | 使用符号计算或自动微分 |

---

## 延伸学习

- **偏微分方程(PDE)**：热传导、波动、扩散方程
- **随机微分方程(SDE)**：加入噪声项
- **延时微分方程(DDE)**：带历史依赖
- **多物种模型**：Lotka-Volterra捕食者-被捕食者
- **网络动力学**：传播在复杂网络上的演化