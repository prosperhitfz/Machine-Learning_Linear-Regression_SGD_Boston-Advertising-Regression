import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def print_regression_expression(coefs):  # 该函数用于给出回归函数的表达式，coefs为回归系数
    names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


# Boston = load_boston()  # 加载Boston房价预测数据集
# print(Boston.DESCR)  # 输出Boston数据集具体内容
# 共506条房屋数据，13个特征变量+1个房屋价格变量
# X = Boston["data"]  # 房屋13种属性
# Y = Boston["target"]  # 房屋均价
AD = pd.read_csv('advertising.csv')
AD = np.array(AD)
X = AD[:, :3]
Y = AD[:, 3]
# print(X)
# print(Y)
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 将数据集做标准化处理后整合到原大小矩阵内
# Y = scaler.fit_transform(Y.reshape(-1, 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)  # 划分训练集和测试集
# 注：这里的random_state用于保证多次程序运行所取的训练集和测试集是一样的，便于观察不同算法对相同数据回归分析的差异


# model = Linear_Regression_Manual()
# Linear_Regression_Manual.fit(X_train, Y_train)

# 线性回归模型拟合数据预测房价
linearegression = LinearRegression()
# 用训练集数据对模型进行训练，得到一条拟合曲线并给出其函数表达式
linearegression.fit(X_train, Y_train)
print("线性回归模型函数表达式: Y = ", print_regression_expression(linearegression.coef_))
# 用经训练集得到的拟合曲线对测试集的数据进行房价预测
Y_linear_predict = linearegression.predict(X_test)
# 计算预测结果对比真实结果的均方根误差
rmse = np.sqrt(np.dot(abs(Y_linear_predict - Y_test), abs(Y_linear_predict - Y_test)) / len(Y_test))
print("预测结果均方根误差: ", rmse)
# 图绘制
plt.plot(Y_test, 'r--', label='real estate price')
plt.plot(Y_linear_predict, 'b-', label='predicted estate price')
plt.title('封装线性回归模型拟合数据预测房价同原房价对比')
plt.legend()
plt.show()

plt.plot(Y_test, Y_linear_predict, 'o')
plt.plot([0, 30], [0, 30], 'y-')
plt.xlabel('real')
plt.ylabel('predicted')
plt.title('封装线性回归模型测试集预测数据（纵轴）同原始数据(横轴）偏差比较')
plt.show()


# 随机梯度下降法拟合数据预测房价
sgd_regression = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)
# 用训练集数据对模型进行训练，得到一条拟合曲线并给出其函数表达式
sgd_regression.fit(X_train, Y_train)
print("随机梯度下降算法函数表达式: Y = ", print_regression_expression(sgd_regression.coef_))
# 用经训练集得到的拟合曲线对测试集的数据进行房价预测
Y_sgd_regression_predict = sgd_regression.predict(X_test)
# 计算分类结果的均方根误差
rmse = np.sqrt(np.dot(abs(Y_sgd_regression_predict - Y_test), abs(Y_sgd_regression_predict - Y_test)) / len(Y_test))
print("预测结果均方根误差: ", rmse)
# 预测房价同实际房价对比图绘制
plt.plot(Y_test, 'r--', label='real estate price')
plt.plot(Y_sgd_regression_predict, 'b-', label='predicted estate price')
plt.title('随机梯度下降法拟合数据预测房价同原房价对比')
plt.legend()
plt.show()

plt.plot(Y_test, Y_sgd_regression_predict, 'o')
plt.plot([0, 30], [0, 30], 'y-')
plt.xlabel('real')
plt.ylabel('predicted')
plt.title('随机梯度下降法测试集预测数据（纵轴）同原始数据(横轴）偏差比较')
plt.show()


