# 📈 预测类算法库 (Prediction & Forecasting Methods)

## 概述 Overview

预测类算法用于**基于历史数据预测未来趋势**，包括时间序列预测、需求预测等。是国赛中最常见的问题类型。

| 算法 | 特点 | 应用场景 | 数据要求 | 难度 |
|------|------|--------|--------|------|
| **线性回归** | 简单快速,可解释性强 | 线性关系预测 | 中等 | ⭐ |
| **ARIMA** | 时间序列标准方法 | 平稳序列预测 | 中等(20+点) | ⭐⭐⭐ |
| **GM(1,1)** | 灰色预测,小样本友好 | 小样本预测 | 少量(4+点) | ⭐⭐ |
| **XGBoost** | 机器学习精准度高 | 复杂非线性关系 | 大量 | ⭐⭐⭐ |
| **LSTM** | 深度学习长期依赖 | 长期序列预测 | 大量 | ⭐⭐⭐⭐ |
| **Prophet** | Facebook开源,自动化 | 含季节性的预测 | 中等 | ⭐⭐ |

---

## 1. 线性回归 (Linear Regression)

### 原理简介

最简单的预测模型：
$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$

通过**最小二乘法**求解参数。

**优点：**
- 简单快速，计算代价小
- 参数有明确统计学解释
- 适合快速建立baseline

**缺点：**
- 假设线性关系
- 对异常值敏感
- 多重共线性问题

### Python实现

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 数据准备
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

# 模型拟合
model = LinearRegression()
model.fit(X, y)

# 预测与评估
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")
print(f"系数: {model.coef_}, 截距: {model.intercept_}")
```

### 改进策略

1. **特征工程**：多项式特征、交互项
2. **正则化**：Ridge(L2)、Lasso(L1)
3. **数据预处理**：标准化、异常值处理

---

## 2. ARIMA (AutoRegressive Integrated Moving Average)

### 原理简介

时间序列预测的标准方法，分解为三部分：
- **AR(p)**：自回归，用过去值预测
- **I(d)**：差分，使序列平稳
- **MA(q)**：移动平均，用过去误差预测

ARIMA(p,d,q) 参数确定：
1. **平稳性检验**：ADF检验，如不平稳则差分（d值）
2. **阶数确定**：ACF/PACF图，或自动搜索最优(p,q)
3. **模型拟合与预测**

### 适用条件

✓ 时间序列数据20个点以上  
✓ 数据具有明显的时间依赖性  
✓ 序列可通过差分变为平稳  

❌ 非平稳趋势数据  
❌ 含有明显的季节性（用SARIMA）

### Python实现

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 时间序列数据
data = np.array([10, 12, 15, 14, 16, 18, 20, 22, 21, 23, 25, 27])

# 平稳性检验 (ADF)
adf_result = adfuller(data)
print(f"ADF统计量: {adf_result[0]:.4f}, p值: {adf_result[1]:.4f}")

# ARIMA模型拟合
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

# 预测
forecast = fitted_model.get_forecast(steps=5)
print(forecast.predicted_mean)

# 模型诊断
fitted_model.plot_diagnostics()
```

### 改进：SARIMA（季节性ARIMA）

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p,d,q)×(P,D,Q,s)，s是季节周期
model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
fitted = model.fit()
```

---

## 3. GM(1,1) 灰色预测模型

### 原理简介

灰色系统论的核心方法，**样本量少(4个点以上即可)** 时仍能建立模型。

**原始数据建立微分方程：**
$$\frac{dx^{(1)}}{dt} + ax^{(1)} = b$$

其中 $x^{(1)}$ 是原数据的一阶累加生成序列。

**解的预测公式：**
$$\hat{x}^{(0)}(k+1) = (x^{(0)}(1) - \frac{b}{a})e^{-ak} + \frac{b}{a}$$

### 优势

✓ 样本量少（国赛常见！）  
✓ 建模周期短  
✓ 预测精度高  

### Python实现

```python
import numpy as np

def gm11_predict(data, predict_steps=5):
    """
    GM(1,1)灰色预测
    
    Args:
        data: 原始时间序列数据 (需4个以上)
        predict_steps: 预测步数
    
    Returns:
        predictions: 预测值序列
    """
    n = len(data)
    
    # 一阶累加生成序列
    x1 = np.cumsum(data)
    
    # 紧邻均值序列
    z1 = np.zeros(n-1)
    for i in range(n-1):
        z1[i] = (x1[i] + x1[i+1]) / 2
    
    # 构造矩阵求解 a, b
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = data[1:].reshape(-1, 1)
    
    params = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = params[0, 0], params[1, 0]
    
    # 预测
    x0_pred = np.zeros(n + predict_steps)
    x1_pred = x1[-1]
    
    for k in range(1, n + predict_steps + 1):
        x1_pred = (data[0] - b/a) * np.exp(-a*k) + b/a
        if k > n:
            x0_pred[k-1] = x1_pred - (x1_pred_prev if k > n+1 else x1[-1])
        x1_pred_prev = x1_pred
    
    return x0_pred[n:]
```

### 改进

- **Verhulst模型**：处理S型曲线
- **参数估计改进**：新息息值法
- **误差修正**：残差修正

---

## 4. XGBoost (eXtreme Gradient Boosting)

### 原理简介

基于梯度提升的机器学习算法，通过**集成多个弱学习器（决策树）** 形成强学习器。

**优点：**
- 精准度高，性能稳定
- 能自动处理非线性关系
- 强大的正则化机制（防过拟合）
- 处理缺失值能力强

**缺点：**
- 需要充足训练数据
- 参数调优复杂
- 可解释性较弱

### Python实现

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

# 准备数据
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100)
X_test = np.random.randn(20, 10)

# 模型配置
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 训练
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
model = xgb.train(params, dtrain, num_boost_round=100)

# 预测
predictions = model.predict(dtest)
```

---

## 5. LSTM (Long Short-Term Memory)

### 原理简介

深度学习中的递归神经网络（RNN），专门处理**长期依赖问题**。

**应用场景：**
- 长时间序列预测
- 有明显季节性的数据
- 数据量充足（1000+）

### Python实现

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 数据准备
data = np.random.randn(200, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 时间步长 = 10
seq_length = 10
X = np.array([scaled_data[i:i+seq_length] for i in range(len(scaled_data)-seq_length)])
y = scaled_data[seq_length:, 0]

# LSTM模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X.reshape(-1, seq_length, 1), y, epochs=50, batch_size=16)

# 预测
predictions = model.predict(X[-1].reshape(1, seq_length, 1))
```

---

## 6. Prophet (Facebook开源库)

### 原理简介

自动分解时间序列为：**趋势 + 季节性 + 假期 + 误差**

特点：**参数少，自动化程度高**，适合快速建模。

```python
from prophet import Prophet
import pandas as pd

# 数据准备 (必须包含 ds 和 y 列)
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365),
    'y': np.random.randn(365).cumsum()
})

# 模型拟合
model = Prophet()
model.fit(df)

# 预测（指定未来30天）
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 绘图
model.plot(forecast)
```

---

## 算法对比与选择

| 算法 | 样本需求 | 精准度 | 实现难度 | 国赛频率 |
|------|--------|------|--------|--------|
| 线性回归 | 少 | 低 | ⭐ | ⭐⭐⭐ |
| ARIMA | 中 | 中 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| GM(1,1) | **很少** | 高 | ⭐⭐ | ⭐⭐⭐ |
| XGBoost | 多 | 高 | ⭐⭐⭐ | ⭐⭐⭐ |
| LSTM | **很多** | 很高 | ⭐⭐⭐⭐ | ⭐⭐ |
| Prophet | 中 | 中 | ⭐ | ⭐⭐ |

---

## 标准预测工作流

```python
class TimeSeriesPredictor:
    """时间序列预测框架"""
    
    def __init__(self, data, train_ratio=0.8):
        self.data = data
        self.n_train = int(len(data) * train_ratio)
        self.train_data = data[:self.n_train]
        self.test_data = data[self.n_train:]
    
    def preprocess(self):
        """数据预处理"""
        # 平稳性检验
        # 异常值处理
        # 数据标准化
        pass
    
    def fit_multiple_models(self):
        """拟合多个模型"""
        self.models = {
            'linear': self.fit_linear(),
            'arima': self.fit_arima(),
            'gm11': self.fit_gm11()
        }
        return self.models
    
    def evaluate(self):
        """模型评估"""
        metrics = {}
        for name, model in self.models.items():
            y_pred = model.predict(len(self.test_data))
            mae = np.mean(np.abs(y_pred - self.test_data))
            rmse = np.sqrt(np.mean((y_pred - self.test_data)**2))
            metrics[name] = {'MAE': mae, 'RMSE': rmse}
        return metrics
```

---

## 评估指标

- **MAE (Mean Absolute Error)**: $\frac{1}{n}\sum|y_i - \hat{y}_i|$
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$
- **MAPE (Mean Absolute Percentage Error)**: $\frac{1}{n}\sum|\frac{y_i - \hat{y}_i}{y_i}|$
- **R²**: 决定系数，越接近1越好

---

## 论文输出形式

1. **预测曲线图**：历史数据 + 预测值 + 置信区间
2. **误差表**：各模型的MAE、RMSE、MAPE对比
3. **残差诊断图**：检验模型假设是否满足
4. **敏感性分析**：参数变化对预测的影响

---

## 常见陷阱

❌ **数据泄露**：用测试集调参  
❌ **过度拟合**：复杂模型拟合噪声  
❌ **忽视季节性**：未进行差分或使用SARIMA  
❌ **评估不公平**：不同模型使用不同验证集  

✓ **最佳实践**：多模型对比、交叉验证、留出验证集
