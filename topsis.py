import pandas as pd
import numpy as np

ids = ['A','B','C','D','E'] # 学校编号
data_monograph = np.array([0.1,0.2,0.4,0.9,1.2]) # 人均专著
data_student_faculty_ratio = np.array([5,6,7,10,2]) # 生师比
data_funds = np.array([5000,6000,7000,10000,400]) # 科研经费
data_graduation = np.array([4.7,5.6,6.7,2.3,1.8]) # 预期毕业率

def normalization1(li):
    y_list = []
    for x in li:
        if x <= 2:
            y_list.append(0)
        elif x > 2 and x <= 5:
            y_list.append((x-2)*1/3)
        elif x > 5 and x <= 6:
            y_list.append(1)
        elif x > 6 and x <= 12:
            y_list.append(1-(x-6)*1/6)
        elif x > 12:
            y_list.append(0)
    return np.array(y_list)

# 生师比数据为区间型指标
data_student_faculty_ratio = normalization1(data_student_faculty_ratio)

# 逾期毕业率为极小型指标，需要进行倒数处理
data_graduation = 1 / data_graduation

# 归一化处理
data_monograph = data_monograph / np.sqrt((data_monograph**2).sum())
data_student_faculty_ratio = data_student_faculty_ratio / np.sqrt((data_student_faculty_ratio**2).sum())
data_funds = data_funds / np.sqrt((data_funds**2).sum())
data_graduation = data_graduation / np.sqrt((data_graduation**2).sum())

# 最优最劣方案
D_minus = []
D_plus = []

for var in [data_monograph, data_student_faculty_ratio, data_funds, data_graduation]:
    D_minus.append(min(var))
    D_plus.append(max(var))

weight = [0.2, 0.3, 0.4, 0.1]
score = []

for i in range(len(ids)):
    # 计算到最优方案的距离
    Di_minus = np.sqrt(
        (data_monograph[i] - D_minus[0])**2 * weight[0] +
        (data_student_faculty_ratio[i] - D_minus[1])**2 * weight[1] +
        (data_funds[i] - D_minus[2])**2 * weight[2] +
        (data_graduation[i] - D_minus[3])**2 * weight[3]
    )
    
    # 计算到最劣方案的距离
    Di_plus = np.sqrt(
        (data_monograph[i] - D_plus[0])**2 * weight[0] +
        (data_student_faculty_ratio[i] - D_plus[1])**2 * weight[1] +
        (data_funds[i] - D_plus[2])**2 * weight[2] +
        (data_graduation[i] - D_plus[3])**2 * weight[3]
    )
    
    # 计算得分并保留4位小数
    score.append(round(Di_minus / (Di_plus + Di_minus), 4))

print("各学校得分:", score)
