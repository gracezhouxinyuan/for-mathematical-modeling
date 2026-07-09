# 稳定性分析（2026-07-09）

```text
stability_analysis/
├── README.md
├── stability_analysis_baseline.py
├── stability_analysis_improved.py
├── plot_stability_analysis_figures.py
├── dataset_sources.md
├── stability_analysis_config.csv
├── results_baseline_eigenvalues.csv
└── results_improved_stability.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：稳定性分析（Stability Analysis）
- 核心思想：通过求解微分方程系统的平衡点，计算 Jacobian 矩阵的特征值，判断系统在平衡点附近的稳定性
- 核心概念：
  - 平衡点：使 f(y*) = 0 的状态点 y*
  - Jacobian 矩阵：J_ij = ∂f_i / ∂y_j，描述系统在平衡点附近的线性近似
  - 线性化：dz/dt = J(y*) * z，将非线性系统在平衡点附近近似为线性系统
  - 特征值判定：Jacobian 矩阵的所有特征值实部为负 → 渐近稳定；存在正实部特征值 → 不稳定
- 适用条件：
  - 微分方程系统存在平衡点
  - 需要判断系统长期行为是否收敛到平衡态
  - 需要分析参数变化对系统稳定性的影响
- 常见失败场景：
  - 平衡点求解错误（数值方法未收敛）
  - Jacobian 矩阵数值微分步长不当导致特征值计算不准
  - 非双曲平衡点（特征值实部为零）需用其他方法分析
  - 多平衡点系统遗漏某些平衡点

### 稳定性分析的改良方法

| 改良方法 | 如何改良 | 相比原版的优点 | 相比原版的缺点 | 适用场景/数据集 |
| -------- | -------- | -------------- | -------------- | --------------- |
| 分岔分析 | 扫描参数范围，追踪平衡点和特征值随参数的变化 | 可发现分岔点和稳定性突变 | 计算量大，需逐参数求解 | 参数敏感的动态系统 |
| 相平面分析 | 绘制相轨迹和方向场 | 直观展示系统全局行为 | 仅适用于二维系统 | 二维 ODE 系统的定性分析 |
| Lyapunov 函数法 | 构造能量函数判定稳定性 | 不依赖线性化，可判定大范围稳定性 | Lyapunov 函数构造困难 | 非线性系统大范围稳定性 |
| 符号计算 Jacobian | 用 SymPy 符号微分代替数值微分 | 特征值精确无误差 | 复杂系统符号计算可能很慢 | 中小规模精确分析 |
| 蒙特卡洛稳定性扫描 | 随机采样参数空间，统计稳定比例 | 可量化参数不确定性下的稳定概率 | 需要大量采样 | 参数不确定的鲁棒性分析 |

**使用建议**

1. 先求平衡点（解析法或数值求根），再计算 Jacobian 特征值。
2. 对于 SIR/SEIR 等传染病模型，重点分析无病平衡点和地方病平衡点。
3. 如果参数有不确定性，用分岔分析或蒙特卡洛扫描评估鲁棒性。
4. 二维系统优先绘制相平面图，直观展示轨迹收敛性。
5. 国赛中稳定性分析常用于传染病模型、生态模型和经济动态模型。

### B. 代码

- baseline 文件：`stability_analysis_baseline.py`
- 改良文件：`stability_analysis_improved.py`
- 可视化文件：`plot_stability_analysis_figures.py`
- 可复现性：`Y`（固定随机种子与固定流程）

### C. 实验

- 指标：特征值实部、稳定性判定结果、分岔图
- 对比对象：
  - baseline：SIR 模型无病平衡点的 Jacobian 特征值分析
  - improved：参数扫描 + 分岔分析 + 相平面图
- 结论目标：说明 R₀ 和感染率 β 对系统稳定性的影响。

### D. 论文

- 方法段：

> 本文针对传染病传播动力学模型进行稳定性分析。首先求解 SIR 模型的无病平衡点和地方病平衡点，随后在平衡点处计算 Jacobian 矩阵并求解其特征值，以判断系统在平衡态附近的局部渐近稳定性。考虑到模型参数（如感染率 β、恢复率 γ）存在不确定性，进一步对关键参数进行扫描，绘制分岔图以展示平衡点稳定性的变化规律。实验结果表明，当基本再生数 R₀ < 1 时无病平衡点渐近稳定，当 R₀ > 1 时无病平衡点失稳且地方病平衡点稳定，该结论为国赛中的传染病防控策略制定提供了理论依据。

---

## 2. 稳定性分析理论补充

稳定性分析是动力系统理论的核心工具，用于判断系统在受到扰动后是否能回到平衡态。

### 核心步骤

1. **求平衡点**：解方程组 f(y*) = 0
2. **计算 Jacobian**：J_ij = ∂f_i / ∂y_j（解析或数值微分）
3. **求特征值**：解 det(J - λI) = 0
4. **判定稳定性**：
   - 所有特征值实部 < 0 → 渐近稳定
   - 存在特征值实部 > 0 → 不稳定
   - 特征值实部 = 0（其余 ≤ 0）→ 临界（需用其他方法）

### SIR 模型的稳定性分析

SIR 模型：
- dS/dt = -βSI/N
- dI/dt = βSI/N - γI
- dR/dt = γI

平衡点：
- 无病平衡点：(S*, I*, R*) = (N, 0, 0)
- 地方病平衡点：当 R₀ = β/γ > 1 时存在

Jacobian 在无病平衡点的特征值：λ₁ = 0, λ₂ = β - γ = γ(R₀ - 1)
- R₀ < 1 (β < γ)：λ₂ < 0 → 稳定
- R₀ > 1 (β > γ)：λ₂ > 0 → 不稳定

### 优点

- 理论严谨，有明确的数学判据
- 可解析推导（对简单系统）
- 能预测系统长期行为

### 缺点

- 仅适用于局部稳定性（平衡点附近）
- 非线性效应强时线性化可能失效
- 高维系统特征值计算困难
- 非双曲平衡点需特殊处理

### 适合场景

- 传染病模型的平衡态分析
- 生态模型（捕食者-被捕食者）的稳定性
- 经济动态模型的均衡分析
- 控制系统的稳定性判定

---

## 3. 运行方式

```bash
python3 stability_analysis/stability_analysis_baseline.py
python3 stability_analysis/stability_analysis_improved.py
python3 stability_analysis/plot_stability_analysis_figures.py
```

运行后你会得到：

- 终端输出：特征值与稳定性判定结果
- 文件输出：
  - `results_baseline_eigenvalues.csv`
  - `results_improved_stability.csv`

---

## 4. 参考来源

- Strogatz, *Nonlinear Dynamics and Chaos*（动力系统经典教材）
- NumPy linalg.eig 官方文档：<https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html>
- SciPy optimize.root：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>
- SIR 模型稳定性分析：<https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>
