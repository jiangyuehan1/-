# 线性回归
# 引入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


"""
引入数据
检查数据
填补数据 
"""
def data_loading_observation(path):
    data=pd.read_csv(path)
    # print(data.describe()) # 查看数据基本情况

    # print(data.isnull().sum())  # 查看数据是否有缺失

    # data[data['列名'].isnull()]=0 # 直接将数据缺失值填补为0
    # data[data['列名']==np.nan]=0
    # data[data['列名']==None]=0
    # data[data['列名'].isnull()]=data['列名'].sum()/data['列名'].count() # 平均值填充
    # data=data.to_csv('new_data') # 将填充后数据重新生成新数据
    return data
"""
绘制图形
"""
def paint(data):
    x=data['TV']
    y=data['sales']
    plt.scatter(x,y)
    plt.show()


"""
线性回归
"""
def db_xxhg(data):

    x_train, x_test, y_train, y_test = train_test_split( data.iloc[:, 1:4],
    data.iloc[:, 4:5], test_size=0.2, random_state=0)

    # x_train = x_train.values.reshape(-1, 1)
    # X_test = x_test.values.reshape(-1, 1)

    model=LinearRegression()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    print("b=" , model.intercept_)
    print("a=" , model.coef_)
    print(model.score(x_test,y_test)) # 回归检验

    plt.plot(range(len(predicted)), predicted, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_test)), y_test, 'green', label="test data")

    plt.show()
    # print(y.shape)

def hgjy(data):
    # 输出相关系数，判断是否值得做线性回归模型
    print(data.corr())  # 0-0.3弱相关；0.3-0.6中相关；0.6-1强相关；

    # 通过seaborn添加一条最佳拟合直线和95%的置信带，直观判断相关关系
    sns.pairplot(data, x_vars=['TV', 'radio','newspaper'], y_vars='sales', size=7, aspect=0.8, kind='reg')
    plt.show()


def yy_zxecf(data):
    """
    一元最小二乘法
    """
    x=data['TV']
    y=data['sales']
    x_pj=sum(x)/len(x)
    y_pj=sum(y)/len(y)
    k=sum((x-x_pj)*(y-y_pj))/sum((x-x_pj)**2)
    b=y_pj-k*x_pj
    return k,b

def dy_zxecf(data):
    """
    多元最小二乘法
    """
    X=data[['TV','radio','newspaper']]
    Y=data['sales']
    X=np.array(X)
    Y=np.array(Y)
    k=((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(Y)

    print(k)


if __name__ == '__main__':
    path = 'D:\pycode\数据科学导引\作业1--线性回归\Advertising.csv'
    data = data_loading_observation(path)

    # paint(data)
    # hgjy(data)
    # db_xxhg(data)
    # yy_zxecf(data)
    dy_zxecf(data)





