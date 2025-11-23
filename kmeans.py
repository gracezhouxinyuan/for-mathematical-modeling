from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.array([
    [9.315, 10.697], [9.213, 9.206], [10.169, 10.504], [8.979, 7.587], 
    [11.042, 10.114], [9.881, 10.628], [9.797, 9.972], [-9.679, -11.054], 
    [-10.580, -12.990], [-9.773, -8.272], [-9.431, -7.546], [-9.139, -7.183], 
    [-9.474, -8.812], [-9.232, -7.298], [-10.478, -10.300], [10.505, -8.092], 
    [7.818, -17.273], [9.385, -11.552], [11.661, -4.377], [11.623, -2.810], 
    [8.648, -15.694], [8.825, -15.968], [10.875, -5.988], [10.945, -7.349]
])

# 挑选出两个维度作为横轴和纵轴
x_axis = data[:, 0]  # 第一列
y_axis = data[:, 1]  # 第二列

# 使用KMeans聚类
model = KMeans(n_clusters=3, random_state=42)
model.fit(data)  # 使用自定义数据训练模型
all_predictions = model.predict(data)  # 预测全部数据

# 绘制聚类结果
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering Results')
plt.show()

# 可选：绘制聚类中心
centers = model.cluster_centers_
plt.scatter(x_axis, y_axis, c=all_predictions, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering with Centers')
plt.show()
