# GA 数据与来源说明

## 内置演示数据

- `ga_knapsack_items.csv`：0-1 背包演示数据，可直接用于 GA 求解。

## 官方/经典来源

- Holland 经典著作：*Adaptation in Natural and Artificial Systems*
- GA 概述资料：<https://en.wikipedia.org/wiki/Genetic_algorithm>

## 改良算法参考

- 自适应变异：<https://arxiv.org/abs/2104.08842>
- 多种群 GA：<https://repository.bilkent.edu.tr/items/a0838796-5471-4995-ac9a-22218cd6eec2>
- 自适应控制参数：<https://www.sciencedirect.com/science/article/pii/S2215098616304736>

## 使用要求

- 本示例默认使用 0-1 背包模型。
- `weight` 与 `value` 均为正数。
- 容量约束写在代码中，适合国赛中快速改造成其他资源约束。
