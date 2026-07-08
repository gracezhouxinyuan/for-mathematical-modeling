# DP 数据与来源说明

## 内置演示数据

- `dp_knapsack_items.csv`：0-1 背包示例数据，用于演示二维 DP 和状态压缩 DP。

## 经典来源

- CLRS: *Introduction to Algorithms* 动态规划章节

## 改良算法参考

- 单调队列优化：<https://cp-algorithms.com/dynamic_programming/knapsack.html>
- 斜率优化 / CHT：<https://pclub.in/tutorial/algos/2016/08/22/dpconv/>
- DP 基础与状态设计：<https://cp-algorithms.com/dynamic_programming/intro-to-dp.html>

## 使用要求

- 本示例默认是 0-1 背包与小规模 TSP。
- 如果状态太多，优先考虑压缩或换建模方式。
