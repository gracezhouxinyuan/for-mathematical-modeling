# TOPSIS 数据与来源说明

## 内置演示数据

- `topsis_demo_data.csv`：可直接跑通 TOPSIS 的方案-指标表。

## 官方公开数据（可替换）

### 1. 国家统计局
- 链接：<https://data.stats.gov.cn/>
- 用法：下载指标数据后，整理成“方案 × 指标”表即可。

### 2. UCI Machine Learning Repository
- 链接：<https://archive.ics.uci.edu/>
- 用法：选择结构化数据集，将类别或数值字段转为综合评价指标。

## 使用要求

- 指标需先区分正向、负向、区间型。
- 进入 TOPSIS 前必须完成无量纲化。
- 如果权重未知，优先使用熵权法或主客观融合权重。
