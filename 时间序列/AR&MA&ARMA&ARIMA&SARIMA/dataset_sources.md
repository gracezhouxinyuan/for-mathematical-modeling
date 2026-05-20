# 时间序列数据来源说明

## 内置数据

- `monthly_series_demo.csv`：月度序列演示数据，可直接跑 AR/MA/ARMA/ARIMA/SARIMA。

## 官方公开数据（可替换）

1. 国家统计局（官方）
- 链接：<https://data.stats.gov.cn/>
- 用法：下载月度/季度指标 CSV 或 Excel，整理为两列：`date,value`。

2. World Bank Data（官方）
- 链接：<https://data.worldbank.org/>
- 用法：选择国家与指标后下载 CSV，按时间排序后提取单一变量序列。

3. OECD Data（官方）
- 链接：<https://data.oecd.org/>
- 用法：下载时间序列指标，保持频率一致（月/季/年）。

## 使用要求

- 时间字段统一为 `YYYY-MM-DD`。
- 不允许重复时间戳。
- 缺失值需先插值或删除。
