from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
    U,S,V = np.linalg.svd(sigma) #奇异分解
    epsilon = 0.1                #白化的时候，防止除数为0
    # ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)   
    # return np.dot(ZCAMatrix, inputs)   #白化变换
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) 
    return np.dot(ZCAMatrix, inputs)                  #计算zca白化矩阵
    

#加载数据集，是一个字典类似Java中的map
Iris = datasets.load_iris()
#print(Iris)

#数据集预处理
Iris_df = pd.DataFrame(Iris.data)
Iris_df = pd.DataFrame(whitening(Iris_df)) 
Iris_df.insert(0,'target',Iris.target)
print(Iris_df)

#按目标分类
Iris_div = []
for i in range(0,3):
    Iris_div.append(Iris_df[Iris_df['target']==i])

#颜色表
colmaps = ['red', 'green', 'blue']

fig_id = 0
for i in range(0,4):
    for j in range(i+1,4):
        fig_id += 1
        plt.subplot(2,3,fig_id)
        for k in range(0,3):
            #挑选出ij两个维度作为x轴和y轴，k作为目标种类
            x_axis = Iris_div[k][i]
            y_axis = Iris_div[k][j]
            #画ij子图的第k种颜色
            plt.scatter(x_axis, y_axis, c=colmaps[k], label=k)
            #添加图例
            plt.legend()


plt.show()