# 卡尔曼滤波器（2026-06-11）

```text
kalman_filter/
├── readme/
│   └── README.md
├── 代码/
│   ├── kalman_filter_baseline.py
│   ├── kalman_filter_improved.py
│   └── plot_kalman_filter_figures.py
├── figure/
└── dataset/
    ├── kalman_demo_data.csv
    ├── dataset_sources.md
    ├── results_baseline_state.csv
    └── results_improved_state.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：卡尔曼滤波器
- 核心思想：利用状态空间模型对系统状态进行递推估计，先预测再用观测修正。
- 核心公式：
  - 状态方程 `x_k = F x_{k-1} + w_k`
  - 观测方程 `z_k = H x_k + v_k`
  - 预测、协方差更新、卡尔曼增益更新
- 适用条件：线性系统、噪声近似高斯、需要在线滤波和状态估计。
- 常见失败场景：模型不线性、噪声统计失真、Q/R 估计不合理、初始值差。

### 卡尔曼滤波器 的改良算法

| 改良方法           | 如何改良                       | 相比原版的优点       | 相比原版的缺点     | 适用场景/数据集          |
| ------------------ | ------------------------------ | -------------------- | ------------------ | ------------------------ |
| 标准卡尔曼滤波     | 固定 Q/R，按线性状态空间递推   | 实现简洁、实时性强   | 参数依赖强         | 线性观测、传感器跟踪     |
| 自适应卡尔曼滤波   | 根据创新序列动态调整噪声协方差 | 对噪声变化更稳       | 规则设计稍复杂     | 噪声强度会变化的时序数据 |
| 扩展卡尔曼滤波 EKF | 对非线性系统做一阶线性化       | 可处理非线性状态方程 | 线性化误差可能较大 | 非线性导航、定位         |
| 无迹卡尔曼滤波 UKF | 用 sigma 点传播均值和协方差    | 比 EKF 更稳健        | 计算略重           | 中等维度非线性系统       |

**使用建议**

1. 基础版适合作为理解问题结构和写论文基线。
2. 当你需要更稳的预测、更平滑的轨迹、更完整的 Pareto 前沿时，优先选改良版。
3. 多目标问题最好把“排序质量 + 稳定性 + 覆盖范围”一起展示。

### B. 代码

- baseline 文件：`代码/kalman_filter_baseline.py`
- 改良文件：`代码/kalman_filter_improved.py`
- 可视化文件：`代码/plot_kalman_filter_figures.py`
- 可复现性：`Y`

### C. 实验

- 指标：请查看 `dataset/results_*.csv` 中保存的核心指标
- 对比对象：baseline vs improved
- 结论目标：在国赛写作中给出“为什么改良版更稳、更准、更适合本题”的解释

### D. 论文沉淀（30分钟）

- 方法段（150-300字）：

> 本文针对含噪时序观测问题构建卡尔曼滤波模型。首先建立状态方程与观测方程，利用系统动力学对下一时刻状态进行预测，再结合观测值计算卡尔曼增益并修正估计结果。基础卡尔曼滤波适用于线性高斯系统，但当观测噪声随时间变化时，固定噪声协方差会导致估计偏差。为提升稳健性，进一步引入基于创新序列的自适应噪声更新机制，使滤波器能够根据当前残差动态调整观测噪声大小。实验结果表明，改良版在轨迹跟踪误差和波动平滑性上更优，适合国赛中的状态估计、传感器融合与在线监测问题。

- 图表编号与说明：
  - `fig_01_kalman_filter_pipeline.png`：卡尔曼滤波器 数据处理与计算流程图
  - `fig_02_kalman_filter_comparison.png`：baseline 与改良版结果对比图
  - `tab_01_kalman_filter_metrics.csv`：核心指标对比表

---

## 2. 理论补充

- 学习算法：卡尔曼滤波器
- 核心思想：利用状态空间模型对系统状态进行递推估计，先预测再用观测修正。
- 核心公式：
  - 状态方程 `x_k = F x_{k-1} + w_k`
  - 观测方程 `z_k = H x_k + v_k`
  - 预测、协方差更新、卡尔曼增益更新
- 适用条件：线性系统、噪声近似高斯、需要在线滤波和状态估计。
- 常见失败场景：模型不线性、噪声统计失真、Q/R 估计不合理、初始值差。

### 数据集选择建议

适合连续状态估计、在线跟踪和传感器融合，尤其适合噪声大但系统规律明确的时序数据。

---

## 3. 运行方式

```bash
python3 kalman_filter/代码/kalman_filter_baseline.py
python3 kalman_filter/代码/kalman_filter_improved.py
python3 kalman_filter/代码/plot_kalman_filter_figures.py
```

---

## 4. 参考来源

- 卡尔曼滤波百科：<https://en.wikipedia.org/wiki/Kalman_filter>
- statsmodels 状态空间文档：<https://www.statsmodels.org/stable/statespace.html>
- 线性动态系统基础：<https://www.youtube.com/watch?v=UnpTlg08sFQ>
