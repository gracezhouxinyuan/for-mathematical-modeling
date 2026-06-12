# KMeans 数据与来源说明

## 内置演示数据

- `kmeans_demo_data.csv`：可直接跑通 KMeans、MiniBatchKMeans 和 BisectingKMeans。

## 官方公开数据（可替换）

### 1. UCI Machine Learning Repository
- 链接：<https://archive.ics.uci.edu/>
- 用法：下载表格型数值数据，标准化后可直接聚类。

### 2. scikit-learn 示例数据集
- 链接：<https://scikit-learn.org/stable/datasets/sample_generators.html>
- 用法：用于快速验证聚类性能和绘图流程。

## 使用要求

- 聚类前必须做标准化。
- 若样本量大，优先 MiniBatchKMeans。
- 若簇数未知，可先用轮廓系数或肘部法选 k。
