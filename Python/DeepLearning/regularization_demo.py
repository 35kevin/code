"""
========================================================
正则化实践案例
========================================================

本案例演示：

1. 什么是过拟合
2. L1 正则化（Lasso）
3. L2 正则化（Ridge）
4. L1 + L2 正则化（ElasticNet）
5. 正则化如何让模型更加平滑

========================================================
"""

# =========================
# 导入库
# =========================

import numpy as np
import matplotlib.pyplot as plt

# 用于生成高阶多项式特征
from sklearn.preprocessing import PolynomialFeatures

# 线性回归模型
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# 自动串联处理流程
from sklearn.pipeline import make_pipeline

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 计算均方误差
from sklearn.metrics import mean_squared_error


# ======================================================
# 1. 生成数据
# ======================================================

"""
真实规律其实很简单：

y = 3x + 2 + 噪声

但是我们后面会用“15阶多项式”去拟合它。

这会导致：
模型复杂度远大于真实问题复杂度

=> 极易过拟合
"""

# 固定随机种子
np.random.seed(42)

# 生成30个点
X = np.linspace(0, 10, 30)

# 加入随机噪声
y = 3 * X + 2 + np.random.randn(30) * 2

# sklearn要求二维输入
X = X.reshape(-1, 1)


# ======================================================
# 2. 划分训练集 / 测试集
# ======================================================

"""
训练集：
用于训练模型

测试集：
用于检验模型泛化能力
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ======================================================
# 3. 设置多项式阶数
# ======================================================

"""
15阶已经非常复杂了。

这种复杂模型：
非常容易记忆训练数据
"""

degree = 15


# ======================================================
# 4. 创建模型
# ======================================================

"""
make_pipeline 的作用：

输入数据
   ↓
PolynomialFeatures
   ↓
回归模型
   ↓
输出结果

=====================================================
"""

# ------------------------------------------------------
# 4.1 普通线性回归（无正则化）
# ------------------------------------------------------

linear_model = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)


# ------------------------------------------------------
# 4.2 L2 正则化（Ridge）
# ------------------------------------------------------

"""
alpha:
正则化强度

越大：
约束越强
"""

ridge_model = make_pipeline(
    PolynomialFeatures(degree),
    Ridge(alpha=1.0)
)


# ------------------------------------------------------
# 4.3 L1 正则化（Lasso）
# ------------------------------------------------------

"""
L1 的特点：

会把某些权重直接压缩成 0

因此：
它可以做“特征选择”
"""

lasso_model = make_pipeline(
    PolynomialFeatures(degree),
    Lasso(alpha=0.01, max_iter=10000)
)


# ------------------------------------------------------
# 4.4 L1 + L2（ElasticNet）
# ------------------------------------------------------

"""
l1_ratio:

=1  -> 纯L1
=0  -> 纯L2

0.5 表示各占一半
"""

elastic_model = make_pipeline(
    PolynomialFeatures(degree),
    ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000
    )
)


# ======================================================
# 5. 训练模型
# ======================================================

linear_model.fit(X_train, y_train)

ridge_model.fit(X_train, y_train)

lasso_model.fit(X_train, y_train)

elastic_model.fit(X_train, y_train)


# ======================================================
# 6. 定义评估函数
# ======================================================

def evaluate_model(model, name):
    """
    计算：
    训练误差
    测试误差
    """

    # 训练集预测
    y_train_pred = model.predict(X_train)

    # 测试集预测
    y_test_pred = model.predict(X_test)

    # 均方误差
    train_error = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_error = mean_squared_error(
        y_test,
        y_test_pred
    )

    print(f"\n{name}")
    print("-" * 40)

    print(f"训练误差: {train_error:.4f}")
    print(f"测试误差: {test_error:.4f}")


# ======================================================
# 7. 输出误差
# ======================================================

evaluate_model(
    linear_model,
    "无正则化"
)

evaluate_model(
    ridge_model,
    "L2 正则化 Ridge"
)

evaluate_model(
    lasso_model,
    "L1 正则化 Lasso"
)

evaluate_model(
    elastic_model,
    "L1 + L2 ElasticNet"
)


# ======================================================
# 8. 绘制拟合曲线
# ======================================================

"""
为了让曲线更平滑：

生成更多连续点
"""

X_plot = np.linspace(0, 10, 500).reshape(-1, 1)


# 创建画布
plt.figure(figsize=(12, 8))


# ------------------------------------------------------
# 原始数据点
# ------------------------------------------------------

plt.scatter(
    X,
    y,
    label="Original Data"
)


# ------------------------------------------------------
# 无正则化
# ------------------------------------------------------

plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    label="No Regularization"
)


# ------------------------------------------------------
# Ridge
# ------------------------------------------------------

plt.plot(
    X_plot,
    ridge_model.predict(X_plot),
    label="Ridge (L2)"
)


# ------------------------------------------------------
# Lasso
# ------------------------------------------------------

plt.plot(
    X_plot,
    lasso_model.predict(X_plot),
    label="Lasso (L1)"
)


# ------------------------------------------------------
# ElasticNet
# ------------------------------------------------------

plt.plot(
    X_plot,
    elastic_model.predict(X_plot),
    label="ElasticNet (L1+L2)"
)


# ======================================================
# 9. 图像美化
# ======================================================

plt.title("Regularization Comparison")

plt.xlabel("X")

plt.ylabel("y")

plt.legend()

plt.grid(True)

plt.show()