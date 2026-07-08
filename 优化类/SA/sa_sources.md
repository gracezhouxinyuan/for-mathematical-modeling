# SA 数据与来源说明

## 内置演示数据

- `sa_tsp_cities.csv`：TSP 城市坐标数据，用于模拟退火路径优化。

## 官方/经典来源

- Kirkpatrick, Gelatt, Vecchi, 1983: *Optimization by Simulated Annealing*

## 改良算法参考

- 自适应降温：<https://arxiv.org/abs/2002.06124>
- SA + 局部搜索：<https://www.sciencedirect.com/science/article/abs/pii/S0377221706006370>
- Parallel Tempering / SA 混合：<https://www.cs.odu.edu/~yaohang/publications/AMChybridPTSA.pdf>

## 使用要求

- 本示例默认是 TSP 路径优化。
- 如果是 VRP 或排班问题，可以把路线交换邻域改成插入、逆序或分段交换。
