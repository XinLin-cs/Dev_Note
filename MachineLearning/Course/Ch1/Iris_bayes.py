from sklearn import datasets
import pandas as pd
import numpy as np
import math
from Bayes import Bayes
from sklearn.model_selection import KFold

# 加载数据集，是一个字典类似Java中的map
Iris = datasets.load_iris()
# 数据集预处理
Iris_x = np.array(Iris.data)
Iris_y = np.array(Iris.target)
# print(Iris_x)
# print(Iris_y)


# k折交叉验证
def kcross(k, data_x, data_y):
    # k折划分子集
    kf = KFold(n_splits=k,shuffle=False)
    for train_index,test_index in kf.split(data_x):
        data_train_x = data_x[train_index]
        data_train_y = data_y[train_index]
        data_test_x = data_x[test_index]
        data_test_y = data_y[test_index]
        
        # 三分类贝叶斯分类器
        myBayse = Bayes(3)
        # 精度验证
        myBayse.trainBayes(data_train_x, data_train_y)
        myBayse.test(data_test_x, data_test_y)


# k折交叉验证, k=5
kcross(5, Iris_x, Iris_y)