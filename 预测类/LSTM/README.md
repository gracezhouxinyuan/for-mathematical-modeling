# LSTM（2026-07-05）

```text
lstm/
├── readme/
│   └── README.md
├── 代码/
│   ├── lstm_baseline.py
│   ├── lstm_improved_multilayer.py
│   └── plot_lstm_figures.py
├── figure/
└── dataset/
    ├── lstm_series_demo.csv
    ├── dataset_sources.md
    ├── results_baseline_metrics.csv
    └── results_improved_metrics.csv
```

---

## 1. 今日任务

### A. 理论

- 学习算法：LSTM（长短期记忆网络）
- 核心公式：
  - 遗忘门：`f_t = sigmoid(W_f[h_{t-1}, x_t] + b_f)`
  - 输入门：`i_t = sigmoid(W_i[h_{t-1}, x_t] + b_i)`
  - 细胞状态：`C_t = f_t*C_{t-1} + i_t*Ĉ_t`
  - 输出门：`o_t = sigmoid(W_o[h_{t-1}, x_t] + b_o)`
- 适用条件：
  - 时间序列预测
  - 存在长期依赖或周期结构
  - 样本量相对充足
- 常见失败场景：
  - 数据太短导致过拟合
  - 未做归一化导致训练不稳定
  - 窗口长度与周期结构不匹配
- 算法的改良算法

| 改良方法            | 如何改良                   | 相比原版的优点       | 相比原版的缺点           | 适用场景/数据集        |
| ------------------- | -------------------------- | -------------------- | ------------------------ | ---------------------- |
| 基础 LSTM           | 单层 LSTM 建模序列窗口     | 能捕捉时序依赖       | 小数据容易过拟合         | 单变量时序预测         |
| 多层 LSTM           | 堆叠多层 LSTM 增强表达能力 | 拟合复杂序列更强     | 训练更慢、过拟合风险更高 | 周期与趋势叠加的序列   |
| Dropout LSTM        | 层间加入 dropout           | 降低过拟合           | 可能降低收敛速度         | 样本不大但噪声较多     |
| 梯度裁剪 + SmoothL1 | 限制梯度范数并使用稳健损失 | 训练更稳定、抗异常值 | 参数需要说明             | 存在突发波动的时序数据 |

**<u>使用建议</u>**

1. LSTM 前先做归一化。
2. 窗口长度要结合周期设定。
3. 如果数据很短，ARIMA/SARIMA 往往比 LSTM 更稳。

### B. 代码

- baseline 文件：`代码/lstm_baseline.py`
- 改良文件：`代码/lstm_improved_multilayer.py`
- 可视化文件：`代码/plot_lstm_figures.py`
- 可复现性：`Y`（固定随机种子与固定流程）

### C. 实验

- 指标：MAE、预测曲线
- 对比对象：单层 LSTM vs 多层 Dropout LSTM
- 结论目标：说明窗口长度、层数和稳健训练策略对时序预测效果的影响。

### D. 论文

- 方法段：

> 本文针对非线性时间序列预测问题构建 LSTM 模型。首先对原始序列进行归一化处理，并利用滑动窗口构造监督学习样本；随后通过 LSTM 的遗忘门、输入门和输出门捕捉序列中的长期依赖关系。基础单层 LSTM 能够完成基本预测，但在趋势与周期叠加的序列中表达能力有限。为提升预测稳定性，进一步采用多层 LSTM、Dropout、SmoothL1 损失和梯度裁剪策略。实验结果表明，改良模型在 MAE 和预测曲线贴合度上更优，适合国赛中的客流、销量、能耗和环境指标预测任务。

---

## 2. 如何运行

在项目根目录执行：

```bash
python3 lstm/代码/lstm_baseline.py
python3 lstm/代码/lstm_improved_multilayer.py
python3 lstm/代码/plot_lstm_figures.py
```

运行后你会得到：

- 终端输出：模型核心指标与部分结果预览
- 文件输出：
- `lstm/dataset/results_baseline_metrics.csv`
- `lstm/dataset/results_improved_metrics.csv`
- `lstm/dataset/results_baseline_predictions.csv`
- `lstm/dataset/results_improved_predictions.csv`
- `lstm/figure/fig_01_lstm_forecast.png`

---

## 3. 数据集说明

本包使用合成趋势、周期和噪声时间序列。真实数据建议来自题目附件、国家统计局、气象/交通/能源公开平台，建模前应检查频率、缺失和异常值。

官方公开来源：

- PyTorch LSTM 官方文档：<https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>
- LSTM 概念：<https://en.wikipedia.org/wiki/Long_short-term_memory>
- 国家统计局数据查询平台：<https://data.stats.gov.cn/>
