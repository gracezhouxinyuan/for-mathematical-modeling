# 📊 评价类算法库 (Evaluation & Scoring Methods)

## 概述 Overview

评价类算法用于**对方案、产品、人物等进行评价、排序和打分**。适用于国赛中的决策分析、综合评价等问题。

| 算法                  | 特点                  | 应用场景              | 难度 |
| --------------------- | --------------------- | --------------------- | ---- |
| **AHP（层次分析法）** | 定性主观判断,分层递阶 | 标准体系构建,权重分配 | ⭐    |
| **熵权法**            | 数据驱动,客观赋权     | 多指标综合评价        | ⭐⭐   |
| **TOPSIS**            | 距离法,归一化处理     | 方案最优选择,综合排序 | ⭐⭐   |

---

## 1. AHP（层次分析法）

### 原理简介

将复杂的决策问题分解为多个层次：**目标层 → 准则层 → 方案层**，通过两两比较得到权重，最后加权汇总得到最终得分。

**核心步骤：**

1. 建立层次结构模型
2. 构造判断矩阵（9级比较标度）
3. 一致性检验（CR < 0.1）
4. 权重计算（特征值法）
5. 综合评分

### 适用条件

- 准则数量中等（3-9个较优）
- 专家评价或主观判断
- 问题结构清晰，可分层

### 常见失败场景

❌ 一致性检验不通过 → 需要调整判断矩阵  
❌ 准则过多（>10个） → 易造成复杂度爆炸  
❌ 无专家意见 → 权重纯主观

### Python实现模板

```python
import numpy as np

def ahp_solve(judgment_matrix):
    """
    AHP求解器
    
    Args:
        judgment_matrix: n×n 判断矩阵
    
    Returns:
        weights: n维权重向量
        cr: 一致性比率
    """
    n = judgment_matrix.shape[0]
    
    # 特征值法计算权重
    eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
    max_lambda = np.max(eigenvalues.real)
    weights = np.abs(eigenvectors[:, np.argmax(eigenvalues.real)].real)
    weights = weights / np.sum(weights)
    
    # 一致性指标和比率
    ci = (max_lambda - n) / (n - 1)
    ri_table = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45]
    ri = ri_table[n-1] if n-1 < len(ri_table) else 1.45
    cr = ci / ri
    
    return weights, cr

# 示例：评价3个方案
judgment_matrix = np.array([
    [1, 2, 5],
    [1/2, 1, 2],
    [1/5, 1/2, 1]
])

weights, cr = ahp_solve(judgment_matrix)
print(f"权重: {weights}, 一致性比率: {cr:.4f}")
```

### 论文写作段落

> 层次分析法（AHP）是由美国运筹学家T.L.Saaty提出的定性与定量相结合的决策方法。该方法将复杂的评价问题按层次分解，通过两两比较确定各层次要素的相对权重，再进行加权合成，以得到总体评价结果。本文建立了包含准则层和方案层的三层递阶模型，通过构造判断矩阵并进行一致性检验(CR=0.082<0.1)，最终计算得到各准则的权重分布。

---

## 2. 熵权法

### 原理简介

基于**信息论**的客观赋权方法。信息熵越大，不确定性越高，权重越小。

**核心步骤：**

1. 数据标准化（归一化到 [0,1]）
2. 计算各指标信息熵
3. 计算权重：$w_j = \frac{1 - e_j}{\sum_k(1 - e_k)}$
4. 加权评分

### 适用条件

- 有完整的数据矩阵
- 需要客观的权重分配
- 指标属性各不相同

### Python实现

```python
import numpy as np

def entropy_weight(data_matrix):
    """
    熵权法赋权
    
    Args:
        data_matrix: m×n 数据矩阵 (m个样本，n个指标)
    
    Returns:
        weights: n维权重向量
    """
    m, n = data_matrix.shape
    
    # 数据归一化
    normalized = data_matrix / data_matrix.sum(axis=0)
    
    # 信息熵计算
    entropy = -np.sum(normalized * np.log(normalized + 1e-10), axis=0) / np.log(m)
    
    # 权重计算
    weights = (1 - entropy) / np.sum(1 - entropy)
    
    return weights
```

---

## 3. TOPSIS（优劣距离法）

### 原理简介

通过计算**正理想解和负理想解的距离**，用接近度 $C_i$ 排序方案。

**核心步骤：**

1. 数据标准化（向量归一化）
2. 加权矩阵（乘以权重）
3. 确定正/负理想解
4. 计算到两个理想解的距离
5. 计算接近度 $C_i = \frac{d^-}{d^+ + d^-}$，取值 [0,1]

### 与AHP的组合：熵权-TOPSIS

```python
import numpy as np

def topsis_evaluate(data_matrix, weights):
    """
    TOPSIS评价
    
    Args:
        data_matrix: 标准化后的指标矩阵
        weights: 权重向量
    
    Returns:
        scores: 各方案接近度分数
    """
    m, n = data_matrix.shape
    
    # 加权矩阵
    weighted = data_matrix * weights
    
    # 理想解与反理想解
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    
    # 距离
    d_plus = np.sqrt(np.sum((weighted - ideal_best)**2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - ideal_worst)**2, axis=1))
    
    # 接近度
    scores = d_minus / (d_plus + d_minus + 1e-10)
    
    return scores
```

---

## 实验对比

| 方法   | 优点             | 缺点         | 国赛频率 |
| ------ | ---------------- | ------------ | -------- |
| AHP    | 直观、易操作     | 主观性强     | ⭐⭐⭐⭐     |
| 熵权法 | 客观、数据驱动   | 需完整数据   | ⭐⭐⭐      |
| TOPSIS | 结论清晰、易理解 | 权重设置关键 | ⭐⭐⭐⭐⭐    |

---

## 常见代码模板

### 标准建模流程

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class EvaluationSystem:
    """评价决策系统"""
    
    def __init__(self, criteria_names, scheme_names):
        self.criteria = criteria_names
        self.schemes = scheme_names
        
    def ahp_weights(self, judgment_matrix):
        """AHP权重计算"""
        pass
    
    def entropy_weights(self, data_matrix):
        """熵权法赋权"""
        pass
    
    def topsis_score(self, data_matrix, weights):
        """TOPSIS评分"""
        pass
    
    def final_ranking(self, scores):
        """最终排序"""
        return np.argsort(-scores)

# 使用示例
system = EvaluationSystem(
    criteria_names=['成本', '质量', '效率'],
    scheme_names=['方案A', '方案B', '方案C']
)
```

---

## 输出形式

### 1. 权重分析表

| 准则  | AHP权重 | 熵权法权重 | 平均权重 |
| ----- | ------- | ---------- | -------- |
| 准则1 | 0.350   | 0.380      | 0.365    |
| 准则2 | 0.280   | 0.250      | 0.265    |
| 准则3 | 0.370   | 0.370      | 0.370    |

### 2. 最终排名表

| 方案  | 综合得分 | 排名 |
| ----- | -------- | ---- |
| 方案A | 0.756    | 1    |
| 方案B | 0.612    | 2    |
| 方案C | 0.521    | 3    |

### 3. 可视化输出

- 雷达图：多准则下各方案的表现
- 柱状图：不同方案的综合得分对比
- 热力图：权重矩阵或评分矩阵

---

## 延伸学习

- **VIKOR法**：处理多准则折中问题
- **模糊AHP**：融合模糊理论的AHP
- **证据理论(D-S)**：处理不确定信息
- **ELECTRE**：欧洲比较法
- **灰色关联-TOPSIS**：小样本情况下的综合评价