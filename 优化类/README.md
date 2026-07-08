# 🎯 优化类算法库 (Optimization Methods)

## 概述 Overview

优化类算法用于**求解最优分配、路径规划、资源调度等问题**。国赛中约30%的题目属于优化问题。

| 算法 | 问题类型 | 特点 | 应用场景 | 难度 |
|------|--------|------|--------|------|
| **LP/MILP** | 线性/整数规划 | 精确求解，完全可靠 | 约束明确的优化 | ⭐⭐ |
| **DP** | 动态规划 | 逐步最优，递推求解 | 阶段决策，最优子结构 | ⭐⭐⭐ |
| **最短路** | 图论 | Dijkstra/Floyd | 路径规划，成本最小 | ⭐⭐ |
| **网络流** | 图论 | 最大流/最小费用流 | 运输配送，资源分配 | ⭐⭐⭐ |
| **GA** | 遗传算法 | 启发式，全局搜索 | 复杂非线性，大规模 | ⭐⭐⭐ |
| **PSO** | 粒子群算法 | 启发式，快速收敛 | 连续参数优化 | ⭐⭐⭐ |
| **SA** | 模拟退火 | 启发式，跳出局部最优 | 组合优化，困难问题 | ⭐⭐⭐ |
| **ACO** | 蚁群算法 | 启发式，自适应 | TSP、VRP、路由问题 | ⭐⭐⭐⭐ |

---

## 第一部分：精确优化算法

## 1. 线性规划 (LP) / 整数规划 (MILP)

### 原理简介

在线性约束条件下，求解线性目标函数的最优值。

**标准形式：**
$$\max/\min \quad c^T x$$
$$s.t. \quad Ax \leq b, \quad x \geq 0$$

**求解方法：**
- **线性规划**：单纯形法（Simplex）
- **整数规划**：分支定界法（Branch & Bound）、割平面法

### 适用条件

✓ 目标函数和约束都是线性的  
✓ 变量类型明确（连续或整数）  
✓ 约束数量有限（不超过几百个）

### Python实现

```python
from scipy.optimize import linprog
import numpy as np

# 最小化问题：min c^T x
c = [1, 2, 3]  # 目标函数系数

# 不等式约束：A_ub @ x <= b_ub
A_ub = [[1, 1, 0], [0, 1, 2]]
b_ub = [4, 6]

# 等式约束：A_eq @ x = b_eq (如果有)
A_eq = [[1, 0, 1]]
b_eq = [3]

# 变量界：x >= 0
x_bounds = [(0, None) for _ in c]

# 求解
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds)

print(f"最优值: {result.fun:.4f}")
print(f"最优解: {result.x}")
```

### 整数规划求解 (PuLP库)

```python
from pulp import *

# 定义问题
prob = LpProblem("Production", LpMaximize)

# 定义变量 (整数)
x1 = LpVariable("Product_A", lowBound=0, cat='Integer')
x2 = LpVariable("Product_B", lowBound=0, cat='Integer')

# 目标函数
prob += 3*x1 + 4*x2, "Total_Profit"

# 约束条件
prob += 2*x1 + x2 <= 100, "Material_A"
prob += x1 + 2*x2 <= 80, "Material_B"
prob += x1 >= 0
prob += x2 >= 0

# 求解
prob.solve()

print(f"状态: {LpStatus[prob.status]}")
print(f"最优利润: {value(prob.objective)}")
print(f"产品A产量: {x1.varValue}, 产品B产量: {x2.varValue}")
```

---

## 2. 动态规划 (Dynamic Programming)

### 原理简介

**将问题分解为重叠的子问题，用递推关系逐步求解**。

**三要素：**
1. **最优子结构**：全局最优由子问题最优组成
2. **重叠子问题**：同一子问题出现多次
3. **状态转移方程**：f(n) = min/max{f(i) + cost(i,n)}

### 经典问题

#### 背包问题 (0/1 Knapsack)

给定n件物品和容量W，每件物品有重量w[i]和价值v[i]，求最大价值。

```python
def knapsack_01(weights, values, capacity):
    """0/1背包问题"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                # 选或不选第i件物品
                dp[i][w] = max(
                    dp[i-1][w],  # 不选
                    dp[i-1][w - weights[i-1]] + values[i-1]  # 选
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# 示例
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8
print(f"最大价值: {knapsack_01(weights, values, capacity)}")
```

#### 最长公共子序列 (LCS)

```python
def lcs(text1, text2):
    """最长公共子序列长度"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

---

## 3. 最短路径算法

### Dijkstra算法（单源最短路）

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    """Dijkstra最短路算法"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current_dist > distances[current]:
            continue
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

### Floyd算法（全源最短路）

```python
def floyd(graph):
    """Floyd全源最短路"""
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    
    # 初始化
    for i in range(n):
        dist[i][i] = 0
    for i in range(n):
        for j, w in graph[i]:
            dist[i][j] = w
    
    # 中间节点k
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
```

---

## 4. 网络流算法

### 最大流 (Ford-Fulkerson)

```python
from collections import defaultdict

def max_flow(graph, source, sink):
    """最大流算法"""
    def bfs(source, sink, parent):
        visited = set([source])
        queue = [source]
        
        while queue:
            u = queue.pop(0)
            for v in graph[u]:
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    queue.append(v)
                    parent[v] = u
                    if v == sink:
                        return True
        return False
    
    parent = {}
    max_flow_value = 0
    
    while bfs(source, sink, parent):
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        
        max_flow_value += path_flow
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        
        parent = {}
    
    return max_flow_value
```

---

## 第二部分：启发式优化算法

## 5. 遗传算法 (Genetic Algorithm, GA)

### 原理简介

模拟自然进化过程：**选择 → 交叉 → 变异 → 新一代**

**步骤：**
1. 初始化种群
2. 计算适应度
3. 选择（轮盘赌、锦标赛）
4. 交叉（单点、多点、均匀）
5. 变异（随机改变基因）
6. 重复直到收敛

### Python实现

```python
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=100, generations=200, mutation_rate=0.1):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def evaluate(self, x):
        """适应度函数 (示例：求最大值)"""
        return -x**2 + 10*x  # 目标：最大化
    
    def selection(self, population, fitness):
        """轮盘赌选择"""
        probs = fitness / np.sum(fitness)
        indices = np.random.choice(len(population), size=self.pop_size, p=probs)
        return population[indices]
    
    def crossover(self, parent1, parent2):
        """单点交叉"""
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutate(self, individual):
        """变异"""
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(len(individual))
            individual[idx] = np.random.randn()
        return individual
    
    def run(self):
        """运行GA"""
        population = np.random.randn(self.pop_size, 1)  # 一维问题示例
        
        for gen in range(self.generations):
            # 评估
            fitness = np.array([self.evaluate(ind[0]) for ind in population])
            fitness = fitness - np.min(fitness) + 1e-10  # 确保正数
            
            # 选择
            population = self.selection(population, fitness)
            
            # 交叉与变异
            new_pop = []
            for i in range(0, self.pop_size, 2):
                c1, c2 = self.crossover(population[i], population[i+1])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
            population = np.array(new_pop)
        
        best_idx = np.argmax([self.evaluate(ind[0]) for ind in population])
        return population[best_idx]
```

---

## 6. 粒子群算法 (PSO)

### 原理简介

模拟鸟群觅食：**每个粒子记住自己的最优位置和全局最优位置**

**速度更新：** 
$$v = w \cdot v + c1 \cdot r1 \cdot (pbest - x) + c2 \cdot r2 \cdot (gbest - x)$$

```python
class ParticleSwarmOptimizer:
    def __init__(self, n_particles=30, n_iterations=100):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 认知系数
        self.c2 = 1.5  # 社会系数
    
    def objective(self, x):
        """目标函数"""
        return np.sum(x**2)  # 最小化
    
    def run(self, bounds):
        """PSO求解"""
        n_dim = len(bounds)
        
        # 初始化
        positions = np.random.uniform([b[0] for b in bounds], 
                                      [b[1] for b in bounds], 
                                      (self.n_particles, n_dim))
        velocities = np.zeros((self.n_particles, n_dim))
        
        pbest_positions = positions.copy()
        pbest_values = np.array([self.objective(p) for p in positions])
        
        gbest_idx = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_idx].copy()
        
        # 迭代
        for _ in range(self.n_iterations):
            r1 = np.random.rand(self.n_particles, n_dim)
            r2 = np.random.rand(self.n_particles, n_dim)
            
            velocities = (self.w * velocities +
                         self.c1 * r1 * (pbest_positions - positions) +
                         self.c2 * r2 * (gbest_position - positions))
            
            positions = positions + velocities
            
            # 边界处理
            for i, (low, high) in enumerate(bounds):
                positions[:, i] = np.clip(positions[:, i], low, high)
            
            # 更新最优值
            values = np.array([self.objective(p) for p in positions])
            improved = values < pbest_values
            pbest_positions[improved] = positions[improved]
            pbest_values[improved] = values[improved]
            
            gbest_idx = np.argmin(pbest_values)
            gbest_position = pbest_positions[gbest_idx].copy()
        
        return gbest_position, pbest_values[gbest_idx]
```

---

## 7. 模拟退火 (Simulated Annealing, SA)

### 原理简介

模拟固体加热冷却的过程，以**概率接受差解**，逃脱局部最优。

**接受概率：** 
$$P(\Delta E) = e^{-\Delta E/T}$$

当T高时，易接受差解；T降低时，逐渐只接受好解。

```python
class SimulatedAnnealing:
    def __init__(self, initial_temp=1000, cooling_rate=0.95):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def objective(self, x):
        """目标函数 (最小化)"""
        return x**2 - 10*x
    
    def run(self, initial_solution, bounds):
        """SA求解"""
        current = initial_solution
        current_value = self.objective(current)
        best = current.copy()
        best_value = current_value
        
        temp = self.initial_temp
        
        while temp > 1e-3:
            # 生成邻域解
            neighbor = current + np.random.randn(*current.shape) * temp
            neighbor = np.clip(neighbor, bounds[0], bounds[1])
            neighbor_value = self.objective(neighbor)
            
            # Metropolis准则
            delta = neighbor_value - current_value
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current = neighbor
                current_value = neighbor_value
                
                if current_value < best_value:
                    best = current.copy()
                    best_value = current_value
            
            temp *= self.cooling_rate
        
        return best, best_value
```

---

## 8. 蚁群算法 (ACO)

### 原理简介

模拟蚂蚁觅食中的信息素传递，适合**组合优化**（如TSP）

**信息素更新：**
$$\tau_{ij} = (1 - \rho) \tau_{ij} + \Delta \tau_{ij}$$

其中 $\Delta \tau_{ij} = Q / L$ (Q为常数，L为路径长度)

```python
class AntColonyOptimizer:
    def __init__(self, n_ants=30, n_iterations=100, alpha=1, beta=2, rho=0.1):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素重要度
        self.beta = beta    # 距离重要度
        self.rho = rho      # 蒸发率
    
    def run(self, distance_matrix):
        """TSP求解"""
        n_cities = len(distance_matrix)
        pheromone = np.ones((n_cities, n_cities))
        
        best_path = None
        best_distance = float('inf')
        
        for _ in range(self.n_iterations):
            paths = []
            for _ in range(self.n_ants):
                path = self._construct_path(pheromone, distance_matrix)
                dist = self._calculate_distance(path, distance_matrix)
                paths.append((path, dist))
                
                if dist < best_distance:
                    best_distance = dist
                    best_path = path
            
            # 更新信息素
            pheromone = pheromone * (1 - self.rho)
            for path, dist in paths:
                for i in range(n_cities):
                    j = (i + 1) % n_cities
                    pheromone[path[i], path[j]] += 1 / dist
        
        return best_path, best_distance
    
    def _construct_path(self, pheromone, distance):
        """蚂蚁构造路径"""
        n = len(pheromone)
        path = [0]
        unvisited = set(range(1, n))
        
        while unvisited:
            current = path[-1]
            # 计算转移概率
            probs = np.zeros(n)
            for j in unvisited:
                probs[j] = (pheromone[current, j] ** self.alpha) / (distance[current, j] ** self.beta)
            probs = probs / np.sum(probs)
            
            next_city = np.random.choice(n, p=probs)
            path.append(next_city)
            unvisited.remove(next_city)
        
        return path
    
    def _calculate_distance(self, path, distance_matrix):
        """计算路径总长"""
        total = 0
        for i in range(len(path)):
            j = (i + 1) % len(path)
            total += distance_matrix[path[i], path[j]]
        return total
```

---

## 算法选择指南

| 问题特征 | 推荐算法 | 理由 |
|--------|--------|------|
| 线性约束，整数变量 | MILP | 精确最优解 |
| 阶段决策，子结构最优 | DP | 高效递推 |
| 小规模图，需完整路径 | Dijkstra/Floyd | 快速精确 |
| 流量平衡，成本分配 | 网络流 | 专门设计 |
| 连续参数，全局优化 | PSO/GA | 启发式搜索 |
| 组合爆炸(TSP/VRP) | SA/ACO | 跳出局部最优 |
| 多目标，冲突目标 | NSGA-II (见多目标类) |  |

---

## 论文标准框架

1. **问题描述与数学建模** → 明确决策变量、目标函数、约束
2. **算法选择与改进** → 为什么选这个、有什么改进
3. **参数设置与实验** → baseline + 改良版本对比
4. **结果分析** → 最优方案、敏感性分析
5. **可视化** → 路径图、收敛曲线、对比图表
