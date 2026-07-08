# KNN 数据与来源说明

## 内置演示数据

- `knn_demo_data.csv`：用于演示 KNN 分类、距离加权与半径邻居。

## 官方文档

- KNeighborsClassifier：<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>
- RadiusNeighborsClassifier：<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html>
- Nearest Neighbors 总览：<https://scikit-learn.org/stable/modules/neighbors.html>

## 改良算法参考

- 距离加权 KNN：scikit-learn `weights='distance'`
- 半径邻居：scikit-learn `RadiusNeighborsClassifier`
- 近似最近邻：可结合 KDTree / BallTree / 外部 ANN 索引

## 使用要求

- 数据必须标准化。
- KNN 对异常值较敏感，建议先做清洗。
- 如果维度很高，先考虑降维或特征选择。
