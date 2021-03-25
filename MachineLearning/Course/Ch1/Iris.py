from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


#加载数据集，是一个字典类似Java中的map
Iris = datasets.load_iris()
#print(Iris)


#数据集预处理
Iris_df = pd.DataFrame(Iris.data)
Iris_df.insert(0,'target',Iris.target)
#按目标分类
Iris_div = []
for i in range(0,3):
    Iris_div.append(Iris_df[Iris_df['target']==i])

#颜色表
colmaps = ['red', 'green', 'blue']

for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            #挑选出ij两个维度作为x轴和y轴，k作为目标种类
            x_axis = Iris_div[k][i]
            y_axis = Iris_div[k][j]
            #画ij子图的第k种颜色
            plt.subplot(3,3,3*i+j+1)
            plt.scatter(x_axis, y_axis, c=colmaps[k], label=k)
            #添加图例
            plt.legend()

plt.show()
plt.savefig('/Figure_1.png')