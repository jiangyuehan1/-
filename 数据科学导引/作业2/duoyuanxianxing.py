"""
一 使用Auto数据集进行多元线性回归(Auto 数据集在资料的打他文件夹里)
(1) 做出数据集中所有变量的散点图矩阵
(2) 计算变量之间的相关系数举证. 需排除定性变量name.
(3) 用处了name变量之外的所有变量作为预测变量, mpg作为响应变量, 进行多元线性回归.
(4)输出回归结果, 分析结果:
      (a) 预测变量和响应变量之间有关系吗?
      (b) 哪个预测变量和响应变量在统计意义上具有显著关系
      (c) 车龄(year)变量的系数说明了什么?
(5) 绘制线性回归拟合结果图,分析拟合中的问题, 残差图表明有异常大的离群点吗?
(6) 对预测变量尝试进行不同的变换, 如 对数变换 logX,  X^2,根號 x 再进行回归分析.
二, 请问当矩阵 X 满足什么条件,  (X 的 T 次方 X) 是可逆的, 并给出证明过程.
三, 查阅资料, 深刻理解径向基函数, 并给出一个基于径向基函数的函数拟合例子(Python编程实现).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
"""
散点图矩阵
"""
def paint(data):
    """
    #y=data['mpg']
    #x=data['cylinders']
    # x = data['displacement']
    # x = data['horsepower']
    # x = data['weight']
    # x = data['acceleration']
    # x = data['year']
    # x = data['origin']
    # x = data['name']
    #plt.scatter(x,y)
    #plt.show()
    """
    sns.pairplot(data.iloc[:,:8])
    plt.show()
"""
相关系数矩阵
"""
def Correlation_coefficient_matrix(data):

    """
    # 计算单个变量的相关系数
    x_1 =np.array(data['cylinders'])
    x_2 = data['displacement']
    x_3 = data['horsepower'].astype('float')
    x_4 = data['weight']
    x_5 = data['acceleration']
    x_6 = data['year']
    x_7 = data['origin']
    y=np.array(data['mpg'])
    print(pearsonr(x_6,y))
    """


    data=data.iloc[:,:8]
    print(data.corr())

"""
多元线性回归
"""
def dyxxhg(data):

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:8],
                                                        data.iloc[:, 0:1], test_size=0.2, random_state=0)
    # x_train = x_train.values.reshape(-1, 1)
    # X_test = x_test.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    print("b=", model.intercept_)
    print("a=", model.coef_)
    print(model.score(x_test, y_test))  # 回归检验
    plt.plot(range(len(predicted)), predicted, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_test)), y_test, 'green', label="test data")

    plt.show()
"""
 log（x）的多元线性回归
"""
def dyxxhg_log(data):

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:8],
                                                        data.iloc[:, 0:1], test_size=0.2, random_state=0)
    # x_train = x_train.values.reshape(-1, 1)
    # X_test = x_test.values.reshape(-1, 1)

    model = LinearRegression()
    x_train = np.log(x_train.astype('float'))
    x_test=np.log(x_test.astype('float'))
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    # print(predicted.head())
    print("b=", model.intercept_)
    print("a=", model.coef_)
    print(model.score(x_test, y_test))  # 回归检验

    plt.plot(range(len(predicted)), predicted, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_test)), y_test, 'green', label="test data")
    plt.show()

    """
    x^2的多元线性函数
    """

def dyxxhg_double(data):

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:8],
                                                        data.iloc[:, 0:1], test_size=0.2, random_state=0)
    # x_train = x_train.values.reshape(-1, 1)
    # X_test = x_test.values.reshape(-1, 1)

    model = LinearRegression()
    x_train = (x_train.astype('float'))**2
    x_test=(x_test.astype('float'))**2
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    # print(predicted.head())
    print("b=", model.intercept_)
    print("a=", model.coef_)
    print(model.score(x_test, y_test))  # 回归检验

    plt.plot(range(len(predicted)), predicted, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_test)), y_test, 'green', label="test data")
    plt.show()

    """
    sqrt（x）的多元线性函数
    """

def dyxxhg_sqrt(data):

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:8],
                                                        data.iloc[:, 0:1], test_size=0.2, random_state=0)
    # x_train = x_train.values.reshape(-1, 1)
    # X_test = x_test.values.reshape(-1, 1)

    model = LinearRegression()
    x_train =np.sqrt(x_train.astype('float'))
    x_test=np.sqrt(x_test.astype('float'))
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    # print(predicted.head())
    print("b=", model.intercept_)
    print("a=", model.coef_)
    print(model.score(x_test, y_test))  # 回归检验

    plt.plot(range(len(predicted)), predicted, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_test)), y_test, 'green', label="test data")
    plt.show()

if __name__=='__main__':
    data=pd.read_csv("Auto.csv")
    # print(data.isnull().sum())
    new_data=data[~data.isin(['?'])]
    new_data.dropna(axis=0,how='any',inplace=True)
    new_data.to_csv('new_data.csv')
    # print(new_data.isnull().sum()) #检查有无缺失值
    # paint(data)
    # Correlation_coefficient_matrix(new_data)
    dyxxhg(new_data)
    # dyxxhg_log(new_data)
    # dyxxhg_double(new_data)
    # dyxxhg_sqrt(new_data)