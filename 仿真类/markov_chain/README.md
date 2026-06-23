# 马尔科夫链（2026-06-23）

```text
markov_chain/
├── readme/
│   └── README.md
├── 代码/
│   ├── markov_chain_baseline.py
│   ├── markov_chain_improved_smoothing.py
│   └── plot_markov_chain_figures.py
├── figure/
└── dataset/
    ├── markov_chain_demo_sequence.csv
    ├── dataset_sources.md
    ├── results_baseline_transition.csv
    └── results_improved_transition.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：马尔科夫链
- 核心公式：
  - 马尔科夫性：`P(X_{t+1}=j | X_t=i, history) = P(X_{t+1}=j | X_t=i)`
  - 转移矩阵：`P_ij = n_ij / sum_j n_ij`
  - 多步预测：`π_{t+k} = π_t P^k`
  - 平稳分布：`π = πP`
- 适用条件：
  - 状态离散
  - 下一状态主要由当前状态决定
  - 需要进行状态预测或长期占比分析
- 常见失败场景：
  - 状态划分不合理
  - 样本序列太短导致转移概率不稳定
  - 高阶历史依赖明显但仍用一阶链
- 算法的改良算法

| 改良方法           | 如何改良                           | 相比原版的优点   | 相比原版的缺点   | 适用场景/数据集          |
| ------------------ | ---------------------------------- | ---------------- | ---------------- | ------------------------ |
| 一阶马尔科夫链     | 只用当前状态预测下一状态           | 简单直观、解释强 | 忽略更长历史     | 天气、等级、状态转移     |
| Laplace 平滑       | 给每个转移计数加先验               | 避免零概率       | 会引入轻微偏差   | 样本较少、状态较多       |
| 高阶马尔科夫链     | 用前 k 个状态共同预测下一状态      | 能刻画更长记忆   | 状态组合爆炸     | 行为序列、复杂状态演化   |
| 隐马尔科夫模型 HMM | 假设状态不可直接观测，只观测到信号 | 适合潜在状态识别 | 训练和解释更复杂 | 语音、健康状态、市场状态 |

**<u>使用建议</u>**

1. 先把连续变量合理离散化。
2. 样本少时使用平滑，避免零概率。
3. 如果观测值不是状态本身，可以考虑 HMM。

### B. 代码

- baseline 文件：`代码/markov_chain_baseline.py`
- 改良文件：`代码/markov_chain_improved_smoothing.py`
- 可视化文件：`代码/plot_markov_chain_figures.py`
- 可复现性：`Y`（固定随机种子与固定流程）

### C. 实验

- 指标：转移矩阵、下一状态概率、平稳分布
- 对比对象：原始频率估计 vs Laplace 平滑估计
- 结论目标：说明平滑对小样本转移概率估计的稳定作用。

### D. 论文

- 方法段：

> 本文针对离散状态演化问题构建马尔科夫链模型。首先根据观测序列统计相邻时刻状态转移次数，并归一化得到状态转移矩阵；随后利用当前状态分布与转移矩阵进行下一时刻预测，并进一步求解平稳分布以刻画系统长期状态占比。考虑到样本序列有限时部分转移可能出现零概率，本文引入 Laplace 平滑修正转移计数，从而提升概率估计的稳健性。实验结果表明，改良后的转移矩阵更适合小样本状态预测和长期趋势分析。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 markov_chain/代码/markov_chain_baseline.py
python3 markov_chain/代码/markov_chain_improved_smoothing.py
python3 markov_chain/代码/plot_markov_chain_figures.py
```

运行后你会得到：

- 终端输出：模型核心指标与部分结果预览
- 文件输出：
- `markov_chain/dataset/results_baseline_transition.csv`
- `markov_chain/dataset/results_improved_transition.csv`
- `markov_chain/dataset/results_stationary_distribution.csv`
- `markov_chain/figure/fig_01_markov_chain.png`

---

## 3. 数据集说明

本包使用合成状态序列展示状态转移估计。真实数据可来自天气状态、信用评级、设备状态、健康等级或题目附件中的离散状态记录。

官方公开来源：

- 马尔科夫链概念：<https://en.wikipedia.org/wiki/Markov_chain>
- HMM 概念：<https://en.wikipedia.org/wiki/Hidden_Markov_model>
- 国家统计局数据查询平台：<https://data.stats.gov.cn/>
