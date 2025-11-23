import matplotlib.pyplot as plt
import numpy as np

N = 25000000
p = 0.01
Q = []
Q1 = []
Q2 = []
s1_list = list(range(2, 30))

for s1 in range(2, 30):
    q1 = np.floor(N / s1)
    q2 = np.floor(N / s1) * (1 - (1 - p) ** s1)*s1
    q = q1 + q2
    Q1.append(q1)
    Q2.append(q2)
    Q.append(q)

plt.figure(figsize=(12, 5))
# 四条不同的线
plt.plot(s1_list, Q, '-o', label='Q')      # Q总和线
plt.plot(s1_list, Q1, '-o', label='Q1')    # Q1线
plt.plot(s1_list, Q2, '-o', label='Q2')    # Q2线
plt.hlines(y=25000000, xmin=2, xmax=30, linewidth=4, label='Num')  # 水平参考线

plt.xlabel('#s#')
plt.ylabel('#Q#')
plt.xticks(range(2, 30))
plt.grid()
plt.legend()
plt.show()
