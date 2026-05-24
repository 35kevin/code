import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成数据
np.random.seed(42)

X = np.linspace(0, 10, 30)
y = 3 * X + 2 + np.random.randn(30) * 2

X = X.reshape(-1, 1)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 高阶多项式（15阶）
degree = 15

# 不使用正则化
model = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)

model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 误差
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print("不使用正则化:")
print("训练误差:", train_error)
print("测试误差:", test_error)


# 使用正则化
model = make_pipeline(
    PolynomialFeatures(degree),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 误差
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print("\n使用正则化:")
print("训练误差:", train_error)
print("测试误差:", test_error)