# 如何使用 Python 进行数据清洗和预处理

## 目录

- [1. 为什么数据清洗决定比赛上限](#1-为什么数据清洗决定比赛上限)
- [2. 国赛标准数据处理总流程](#2-国赛标准数据处理总流程)
- [3. 环境与依赖](#3-环境与依赖)
- [4. 一份可复用的预处理主脚本模板](#4-一份可复用的预处理主脚本模板)
- [5. 缺失值处理策略](#5-缺失值处理策略)
- [6. 异常值处理策略](#6-异常值处理策略)
- [7. 重复值与逻辑冲突清洗](#7-重复值与逻辑冲突清洗)
- [8. 类型、单位与时间字段标准化](#8-类型单位与时间字段标准化)
- [9. 编码与文本字段处理](#9-编码与文本字段处理)
- [10. 特征缩放与分布变换](#10-特征缩放与分布变换)
- [11. 类别不平衡处理](#11-类别不平衡处理)
- [12. 特征工程与特征筛选](#12-特征工程与特征筛选)
- [13. 训练/验证划分与防止数据泄漏](#13-训练验证划分与防止数据泄漏)
- [14. 质量评估与自动检查清单](#14-质量评估与自动检查清单)
- [15. 论文写作怎么描述数据清洗](#15-论文写作怎么描述数据清洗)
- [16. 国赛常见坑与应对](#16-国赛常见坑与应对)
- [17. 快速执行版（比赛当天）](#17-快速执行版比赛当天)

---

### 1. 为什么数据清洗决定比赛上限

在国赛中，算法通常不是第一失分点，数据质量才是。  
常见失败模式：

- 指标单位混乱（元/万元、kg/t）导致结果量级错误。
- 异常值未处理导致回归或聚类结果被“带飞”。
- 训练集和测试集信息泄漏，离线精度虚高、解释失真。
- 缺失值粗暴删除后样本结构改变，结论不稳健。

结论：**先把数据做对，再谈模型高分。**

---

### 2. 国赛标准数据处理总流程

1. 数据盘点（字段、单位、来源、时间跨度）
2. 字段类型修正（数值/类别/时间）
3. 缺失值处理
4. 异常值检测与处理
5. 重复值与逻辑冲突清洗
6. 单位统一与尺度标准化
7. 特征工程
8. 划分训练/验证集
9. 保存可复现中间数据
10. 产出清洗报告（可写进论文）

---

### 3. 环境与依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

推荐固定随机种子：

```python
RANDOM_STATE = 42
```

---

### 4. 一份可复用的预处理主脚本模板

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    return df


def impute_missing(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "UNKNOWN")
    return df


def clip_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower, upper)
    return df


def standardize_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def main() -> None:
    df = load_data("data/raw/input.xlsx")
    df = basic_clean(df)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    df = impute_missing(df, num_cols=num_cols, cat_cols=cat_cols)
    df = clip_outliers_iqr(df, cols=num_cols)
    df = standardize_features(df, cols=num_cols)

    df.to_csv("data/processed/cleaned.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
```

---

### 5. 缺失值处理策略

#### 5.1 先判断缺失机制

- MCAR：完全随机缺失
- MAR：条件随机缺失
- MNAR：非随机缺失

#### 5.2 处理建议

- 缺失比例 < 5%：中位数/众数填补通常可接受。
- 5%-30%：优先分组填补、插值、模型填补。
- >30%：评估字段价值，必要时删除该字段。

#### 5.3 国赛推荐

- 时间序列：线性插值、样条插值、前向/后向填充。
- 截面数据：分组中位数填补（按地区、类型、等级）。

---

### 6. 异常值处理策略

#### 6.1 常用方法

- IQR 法：稳健、解释清晰，论文友好。
- 3σ 法：适合近似正态数据。
- 模型法：Isolation Forest/LOF，适合复杂分布。

#### 6.2 国赛实操策略

- 先画箱线图 + 直方图确认分布。
- 再决定是删除、截断（clip）还是 Winsorize。
- 对“极端但真实”的业务值不要盲删。

---

### 7. 重复值与逻辑冲突清洗

#### 7.1 重复值

```python
df = df.drop_duplicates()
```

#### 7.2 主键冲突

```python
# 示例：同一ID同一天出现多条记录
conflict = df.duplicated(subset=["id", "date"], keep=False)
```

#### 7.3 逻辑规则校验

- 年龄 < 0 或 > 120
- 时间先后颠倒（结束时间 < 开始时间）
- 比例值超出 [0, 1]

---

### 8. 类型、单位与时间字段标准化

#### 8.1 类型修正

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

#### 8.2 单位统一

- 金额：元、千元、万元统一到元。
- 重量：g、kg、t统一到kg或t。
- 长度：mm、cm、m统一到m。

#### 8.3 时间特征扩展

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday
```

---

### 9. 编码与文本字段处理

#### 9.1 类别编码

- 类别有序：Label Encoding
- 类别无序：One-Hot Encoding

```python
df = pd.get_dummies(df, columns=["region", "type"], drop_first=True)
```

#### 9.2 文本规范化

- 去空格、统一全角半角、统一大小写
- 同义词归并（例如“北京/北京市”）

---

### 10. 特征缩放与分布变换

#### 10.1 何时标准化

- 距离类算法：KMeans、KNN、SVM 必做。
- 神经网络与梯度优化：建议做。
- 树模型：通常不敏感，可选。

#### 10.2 分布偏态处理

- 对数变换：`log1p`
- Box-Cox / Yeo-Johnson

```python
df["x_log"] = np.log1p(df["x"])
```

---

### 11. 类别不平衡处理

- 欠采样 / 过采样
- SMOTE（合成少数类）
- 阈值调优 + 代价敏感学习

评价指标不要只看 Accuracy，优先看：

- Precision / Recall / F1
- AUC
- 混淆矩阵

---

### 12. 特征工程与特征筛选

#### 12.1 常见特征工程

- 比率特征（如人均、单位产出）
- 交互特征（乘积、差值）
- 分箱特征
- 滞后特征（时间序列）

#### 12.2 常见筛选方式

- 相关性筛选
- 方差筛选
- L1 正则筛选
- 树模型特征重要性

---

### 13. 训练/验证划分与防止数据泄漏

#### 13.1 划分原则

- 普通任务：`train_test_split`
- 时间序列：按时间先后切分，不可随机打散

#### 13.2 泄漏高发点

- 在全量数据上先标准化再切分
- 用测试集参与参数选择
- 用未来信息构造当前特征

正确顺序：**先切分，再在训练集拟合变换器，再应用到验证/测试集。**

---

### 14. 质量评估与自动检查清单

清洗完成后至少输出以下检查项：

- 缺失率统计表（清洗前后）
- 异常值数量变化表
- 关键字段分布对比图（前后）
- 目标变量分布图
- 训练/验证集一致性检查

建议保存文件：

- `data/processed/cleaned.csv`
- `reports/tables/missing_report.csv`
- `reports/figures/distribution_compare.png`
