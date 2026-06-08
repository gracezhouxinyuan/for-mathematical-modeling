# 🎪 多目标优化算法库 (Multi-Objective Optimization)

## 概述 Overview

多目标优化处理**多个相互冲突的目标**。目标是找到**Pareto前沿**（不存在某个解同时在所有目标上都更优）。

| 算法 | 原理 | 特点 | 难度 |
|------|------|------|------|
| **NSGA-II** | 非支配排序+拥挤度 | 经典、高效、应用广 | ⭐⭐⭐ |
| **MOPSO** | 多目标粒子群 | 群体智能，全局搜索 | ⭐⭐⭐ |
| **MOEA/D** | 分解方法 | 参数控制灵活 | ⭐⭐⭐⭐ |
| **NSGA-III** | NSGA-II改进 | 处理多于3个目标 | ⭐⭐⭐⭐ |
| **SPEA2** | 强度Pareto进化 | 精英保留，高精度 | ⭐⭐⭐⭐ |

---

## 1. Pareto最优性基础

### 定义

对于最小化问题，如果**不存在其他解在所有目标上都更优**，则称为Pareto最优解。

**数学定义：** 解 $x^*$ 是Pareto最优的，当且仅当：
- 不存在 $x$ 使得 $f_i(x) \leq f_i(x^*)$ 对所有 $i$ 成立，且至少有一个严格不等号

### Pareto前沿

所有Pareto最优解的集合。

```python
import numpy as np
import matplotlib.pyplot as plt

def is_dominated(sol1, sol2):
    """检查sol1是否被sol2支配"""
    return np.all(sol2 <= sol1) and np.any(sol2 < sol1)

def get_pareto_front(solutions):
    """
    从解集中提取Pareto前沿
    
    Args:
        solutions: n×m数组，n个解，m个目标
    
    Returns:
        indices: Pareto前沿中解的索引
    """
    n = len(solutions)
    is_pareto = [True] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and is_dominated(solutions[i], solutions[j]):
                is_pareto[i] = False
                break
    
    return np.where(is_pareto)[0]

# 示例
np.random.seed(42)
solutions = np.random.randn(100, 2) * [[3, 2]] + [[5, 4]]
pareto_idx = get_pareto_front(solutions)

plt.figure(figsize=(8, 6))
plt.scatter(solutions[:, 0], solutions[:, 1], alpha=0.3, label='Solutions')
plt.scatter(solutions[pareto_idx, 0], solutions[pareto_idx, 1], c='red', s=100, label='Pareto Front')
plt.xlabel('f1 (minimize)')
plt.ylabel('f2 (minimize)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. NSGA-II (Non-dominated Sorting Genetic Algorithm II)

### 原理

**最广泛使用的多目标优化算法**。核心思想：
1. **非支配排序**：将种群分层（第一层Pareto前沿，第二层次优...）
2. **拥挤度距离**：衡量解的多样性（在目标空间中的"孤独度"）
3. **精英保留**：保留非支配解

### Python实现

```python
import numpy as np
from typing import List, Tuple

class NSGAII:
    def __init__(self, obj_func, num_var, bounds, pop_size=100, generations=200):
        """
        Args:
            obj_func: 目标函数，返回m维目标向量
            num_var: 决策变量数
            bounds: [(low, high), ...] 变量界
            pop_size: 种群大小
            generations: 进化代数
        """
        self.obj_func = obj_func
        self.num_var = num_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
    
    def initialize_population(self):
        """初始化种群"""
        pop = np.zeros((self.pop_size, self.num_var))
        for i, (low, high) in enumerate(self.bounds):
            pop[:, i] = np.random.uniform(low, high, self.pop_size)
        return pop
    
    def evaluate(self, population):
        """评估种群"""
        objectives = np.array([self.obj_func(ind) for ind in population])
        return objectives
    
    def non_dominated_sorting(self, objectives):
        """非支配排序"""
        pop_size = len(objectives)
        num_obj = objectives.shape[1]
        
        # 初始化支配关系
        dominates = [[] for _ in range(pop_size)]
        dominated_count = np.zeros(pop_size)
        
        # 计算支配关系
        for i in range(pop_size):
            for j in range(i+1, pop_size):
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    dominates[j].append(i)
                    dominated_count[i] += 1
        
        # 分层
        fronts = []
        current_front = np.where(dominated_count == 0)[0]
        
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current_front = np.unique(next_front)
        
        return fronts
    
    def crowding_distance(self, objectives, front_indices):
        """计算拥挤度距离"""
        n_points = len(front_indices)
        n_obj = objectives.shape[1]
        
        distances = np.zeros(n_points)
        
        for m in range(n_obj):
            # 按第m个目标排序
            sorted_indices = np.argsort(objectives[front_indices, m])
            
            # 边界点距离为无穷
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # 计算其他点的距离
            obj_min = objectives[front_indices[sorted_indices[0]], m]
            obj_max = objectives[front_indices[sorted_indices[-1]], m]
            
            for i in range(1, n_points - 1):
                idx = front_indices[sorted_indices[i]]
                idx_prev = front_indices[sorted_indices[i-1]]
                idx_next = front_indices[sorted_indices[i+1]]
                
                distances[sorted_indices[i]] += (
                    (objectives[idx_next, m] - objectives[idx_prev, m]) /
                    (obj_max - obj_min + 1e-10)
                )
        
        return distances
    
    def selection(self, population, objectives, fronts):
        """选择新种群"""
        new_pop_indices = []
        
        for front in fronts:
            if len(new_pop_indices) + len(front) <= self.pop_size:
                new_pop_indices.extend(front)
            else:
                # 按拥挤度排序选择
                remaining = self.pop_size - len(new_pop_indices)
                distances = self.crowding_distance(objectives, front)
                selected = np.argsort(-distances)[:remaining]  # 拥挤度大的优先
                new_pop_indices.extend(front[selected])
                break
        
        return new_pop_indices
    
    def crossover(self, parent1, parent2):
        """单点交叉"""
        point = np.random.randint(1, self.num_var)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutation(self, individual):
        """高斯变异"""
        mutation_rate = 1 / self.num_var
        for i in range(self.num_var):
            if np.random.rand() < mutation_rate:
                low, high = self.bounds[i]
                individual[i] = np.clip(
                    individual[i] + np.random.normal(0, (high-low)*0.1),
                    low, high
                )
        return individual
    
    def run(self):
        """运行NSGA-II"""
        pop = self.initialize_population()
        all_fronts = []  # 记录每代的前沿
        
        for gen in range(self.generations):
            # 评估
            objectives = self.evaluate(pop)
            
            # 非支配排序
            fronts = self.non_dominated_sorting(objectives)
            
            # 选择
            selected_idx = self.selection(pop, objectives, fronts)
            pop = pop[selected_idx]
            
            # 记录Pareto前沿
            all_fronts.append(objectives[fronts[0]])
            
            # 交叉和变异生成新种群
            new_pop = []
            for _ in range(self.pop_size // 2):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                c1, c2 = self.crossover(pop[i1], pop[i2])
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                new_pop.extend([c1, c2])
            
            pop = np.array(new_pop[:self.pop_size])
            
            if (gen + 1) % 50 == 0:
                print(f"第 {gen+1} 代完成，前沿大小: {len(fronts[0])}")
        
        # 最终Pareto前沿
        objectives = self.evaluate(pop)
        fronts = self.non_dominated_sorting(objectives)
        
        return pop[fronts[0]], objectives[fronts[0]], all_fronts

# 示例：二目标优化问题
# 最小化 f1(x) = x1^2，f2(x) = (x1-2)^2
def zdt_test(x):
    return np.array([x[0]**2, (x[0]-2)**2])

nsga2 = NSGAII(
    obj_func=lambda x: zdt_test(x),
    num_var=1,
    bounds=[(0, 4)],
    pop_size=50,
    generations=100
)

pareto_solutions, pareto_objectives, history = nsga2.run()

# 绘制结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], s=100, c='red', edgecolors='k')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.title('最终Pareto前沿')
plt.grid(True)

plt.subplot(1, 2, 2)
for i, front in enumerate(history[::10]):  # 每10代取一次
    plt.scatter(front[:, 0], front[:, 1], alpha=0.3, s=20)
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.title('优化过程')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 3. MOPSO (多目标粒子群优化)

```python
class MOPSO:
    def __init__(self, obj_func, num_var, bounds, n_particles=30, generations=100):
        self.obj_func = obj_func
        self.num_var = num_var
        self.bounds = bounds
        self.n_particles = n_particles
        self.generations = generations
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 认知系数
        self.c2 = 1.5  # 社会系数
    
    def initialize(self):
        """初始化粒子群"""
        particles = np.zeros((self.n_particles, self.num_var))
        velocities = np.zeros((self.n_particles, self.num_var))
        
        for i, (low, high) in enumerate(self.bounds):
            particles[:, i] = np.random.uniform(low, high, self.n_particles)
            velocities[:, i] = np.random.uniform(-0.5, 0.5, self.n_particles)
        
        return particles, velocities
    
    def run(self):
        """运行MOPSO"""
        particles, velocities = self.initialize()
        objectives = np.array([self.obj_func(p) for p in particles])
        
        # 初始化个人最优和全局最优
        pbest_particles = particles.copy()
        pbest_objectives = objectives.copy()
        
        # 初始化外部存档（Pareto前沿）
        archive = self._get_pareto_front(objectives)
        
        for gen in range(self.generations):
            for i in range(self.n_particles):
                # 从外部存档随机选择一个作为全局最优
                gbest_idx = archive[np.random.randint(len(archive))]
                gbest = pbest_particles[gbest_idx]
                
                # 速度和位置更新
                r1 = np.random.rand(self.num_var)
                r2 = np.random.rand(self.num_var)
                
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (pbest_particles[i] - particles[i]) +
                               self.c2 * r2 * (gbest - particles[i]))
                
                particles[i] = particles[i] + velocities[i]
                
                # 边界处理
                for j, (low, high) in enumerate(self.bounds):
                    particles[i, j] = np.clip(particles[i, j], low, high)
            
            # 评估
            objectives = np.array([self.obj_func(p) for p in particles])
            
            # 更新个人最优
            for i in range(self.n_particles):
                if self._dominates(objectives[i], pbest_objectives[i]):
                    pbest_particles[i] = particles[i].copy()
                    pbest_objectives[i] = objectives[i].copy()
            
            # 更新外部存档
            combined_objs = np.vstack([pbest_objectives, objectives])
            pareto_idx = self._get_pareto_front(combined_objs)
            archive = pareto_idx[pareto_idx < len(pbest_objectives)]
            
            if (gen + 1) % 20 == 0:
                print(f"第 {gen+1} 代，存档大小: {len(archive)}")
        
        return pbest_particles[archive], pbest_objectives[archive]
    
    def _dominates(self, obj1, obj2):
        """检查obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _get_pareto_front(self, objectives):
        """提取Pareto前沿索引"""
        n = len(objectives)
        is_pareto = [True] * n
        
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    is_pareto[i] = False
                    break
        
        return np.where(is_pareto)[0]
```

---

## 4. 多目标性能指标

### Hypervolume (HV)

表示Pareto前沿支配的目标空间体积，**越大越好**。

```python
def hypervolume(objectives, reference_point):
    """
    计算超体积（简化版）
    
    Args:
        objectives: Pareto前沿目标向量
        reference_point: 参考点（通常是目标函数最坏值）
    """
    # 按第一个目标排序
    sorted_idx = np.argsort(objectives[:, 0])
    
    hv = 0
    for i in range(len(objectives)):
        idx = sorted_idx[i]
        width = reference_point[0] - objectives[idx, 0]
        
        if i == 0:
            height = reference_point[1] - objectives[idx, 1]
        else:
            prev_idx = sorted_idx[i-1]
            height = objectives[prev_idx, 1] - objectives[idx, 1]
        
        hv += width * height
    
    return hv

# 示例
reference = np.array([5, 5])
hv = hypervolume(pareto_objectives, reference)
print(f"Hypervolume: {hv:.4f}")
```

### IGD (Inverted Generational Distance)

衡量**获得的Pareto前沿与真实前沿的距离**，越小越好。

```python
def igd(obtained_front, true_front):
    """
    反向代际距离
    
    Args:
        obtained_front: 算法获得的前沿
        true_front: 真实的Pareto前沿
    
    Returns:
        IGD值
    """
    distances = []
    for point in true_front:
        min_dist = np.min([np.linalg.norm(point - p) for p in obtained_front])
        distances.append(min_dist)
    
    return np.mean(distances)
```

---

## 论文输出标准

1. **Pareto前沿图** → 二目标用2D，三目标用3D
2. **收敛性曲线** → Hypervolume随代数的变化
3. **多个算法对比** → NSGA-II vs MOPSO vs ... 的HV对比表
4. **分布多样性分析** → 前沿的分布密度
5. **计算时间对比** → 不同算法的运行时间

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 前沿退化 | 目标函数冲突不足 | 检查目标函数定义 |
| 收敛到局部 | 种群多样性丧失 | 增加种群大小或变异率 |
| 前沿振荡 | 参数设置不稳定 | 降低学习率，增加代数 |
| 计算太慢 | 问题规模过大 | 使用代理模型或降维 |

---

## 延伸学习

- **Many-objective优化**：3个以上目标的优化
- **约束多目标优化(CMOP)**：加入约束条件
- **动态多目标优化(DMOP)**：目标函数随时间变化
- **代理辅助多目标优化**：使用代理模型加速收敛
- **偏好学习**：根据决策者偏好引导搜索
