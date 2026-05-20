# AR/MA/ARMA/ARIMA/SARIMA Family
# 

```text
arima_family/
├── readme/
│   └── README.md
├── 代码/
│   ├── ts_baseline_models.py
│   ├── ts_improved_sarima_check.py
│   └── plot_ts_figures.py
├── figure/
└── dataset/
    ├── monthly_series_demo.csv
    ├── dataset_sources.md
    └── results_metrics.csv
```

---

## 今日任务（2026-05-20）

### A. 理论

- 学习算法：AR / MA / ARMA / ARIMA / SARIMA
- 核心公式：
  - AR(p): `y_t = c + Σ(phi_i y_{t-i}) + eps_t`
  - MA(q): `y_t = mu + eps_t + Σ(theta_j eps_{t-j})`
  - ARMA(p,q): AR + MA（平稳序列）
  - ARIMA(p,d,q): 差分后做 ARMA（非平稳序列）
  - SARIMA(p,d,q)(P,D,Q,s): 在 ARIMA 中加入季节项
- 适用条件：
  - 单变量时间序列预测
  - 数据存在自相关
  - 样本中等规模，强调可解释性
- 常见失败场景：
  - 未检验平稳性直接建 ARMA
  - 季节性明显却不用 SARIMA
  - 只看拟合误差，不看残差白噪声

### B. 代码

- baseline 文件：`代码/ts_baseline_models.py`
- 改良文件：`代码/ts_improved_sarima_check.py`
- 可视化文件：`代码/plot_ts_figures.py`
- 是否可复现（Y/N）：Y（固定数据与参数）

### C. 实验

- 指标：MAE、RMSE、MAPE
- 对比对象：AR / MA / ARMA / ARIMA / SARIMA
- 结论目标：验证季节序列场景下 SARIMA 的优势

### D. 论文

- 方法段：

> 本文针对单变量时间序列预测任务，构建 AR、MA、ARMA、ARIMA 与 SARIMA 多模型对比框架。首先通过时间序列可视化与自相关特征识别其趋势与季节性，并在训练集上分别完成模型参数设定与拟合。考虑到原序列存在季节波动，进一步引入季节项构建 SARIMA 模型。随后在统一测试集上采用 MAE、RMSE 与 MAPE 指标评价模型性能，并结合残差诊断检验模型有效性。实验结果表明，SARIMA 在季节性序列上的预测误差最低，说明加入季节结构可显著提升预测精度与稳定性。

- 图表编号与说明：
  - `fig_01_series_train_test.png`：训练/测试集拆分图
  - `fig_02_forecast_comparison.png`：五模型预测对比图
  - `fig_03_residual_diagnostics.png`：改良模型残差诊断图
  - `results_metrics.csv`：各模型误差指标对比表

---

## 如何运行

```bash
python3 arima_family/代码/ts_baseline_models.py
python3 arima_family/代码/ts_improved_sarima_check.py
python3 arima_family/代码/plot_ts_figures.py
```

---

## 数据说明
- 官方数据来源与下载说明见：`dataset/dataset_sources.md`。
