import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

learning_rate = 0.01  # 学习率（步长）
max_iterations = 5000  # 迭代次数
l = 50  # 正则化参数


def Manual_Linear_Regression(x, y, w, l, alpha, iteration):  # 梯度下降
    loss = np.zeros(iteration)  # 初始化一个数组，用于储存每代的损失
    for i in range(iteration):
        # 利用向量化一步求解，w为权重
        w = w - (alpha / x.shape[0]) * (x * w.T - y).T * x - (alpha * l / x.shape[0]) * w  # 添加了正则项
        loss_mid = np.power(x * w.T - y, 2)  # (wx-y)^2
        reg = (l / (2 * len(x))) * (np.power(w, 2).sum())  # 正则化项
        loss[i] = np.sum(loss_mid)/(2 * len(x)) + reg      # 记录每次迭代后的损失函数值=均值+正则项
        if i % 100 == 0:
            print('第', i, '次迭代的loss:', loss[i])
    return w, loss


data = pd.read_csv('advertising.csv')
data = (data - data.mean())/data.std()  # 特征缩放: （x-平均值）/标准差
# print(data.head(5))  # 查看特征缩放后的数据
cols = data.shape[1]  # 列数 shape[0]行数 [1]列数
X = data.iloc[:, 0:cols-1]  # 取前cols-1列，即特征
Y = data.iloc[:, cols - 1:cols]  # 取最后一列，即价格（拟合目标）
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
# 将数据转换成numpy矩阵
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)
# 初始化偏置矩阵w
w = np.matrix([0, 0, 0, 0])
# 添加偏置列，值为1（axis = 1表示添加列）
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)

# 运行线性回归算法用训练集拟合出最终的w值 代价函数J(theta)
final_theta, cost = Manual_Linear_Regression(X_train, y_train, w, l, learning_rate, max_iterations)
# print("手写线性回归模型函数表达式: Y = ", final_theta)
final_theta = np.array(final_theta)
print("手写线性回归模型函数表达式: Y = ", final_theta[0][3], '+',
      final_theta[0][0], '* X1 +',
      final_theta[0][1], '* X2 +',
      final_theta[0][2], '* X3')
# 模型评估（均方根计算）
y_test_predict = X_test * final_theta.T
rmse = np.sqrt(np.sum(np.power(y_test_predict - y_test, 2)) / (len(X_test)))
print('预测结果均方根误差： ', rmse)

# 图例展示预测值与真实值的变化趋势
plt.plot(y_test, 'r--', linewidth=2, label='real')
plt.plot(y_test_predict, 'b-', linewidth=2, label='predicted')
plt.legend(loc='upper right')
plt.title('手写线性回归预测销量')
plt.show()

plt.plot(y_test, y_test_predict, 'o')
plt.plot([-3, 3], [-3, 3], 'y-')
plt.xlabel('real')
plt.ylabel('predicted')
plt.title('手写线性回归法测试集预测数据（纵轴）同原始数据(横轴）偏差比较')
plt.show()
