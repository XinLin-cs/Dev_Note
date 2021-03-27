from sklearn import datasets
import pandas as pd
import math
from MED import MED

if __name__ == '__main__':
    #加载数据集，是一个字典类似Java中的map
    Iris = datasets.load_iris()
    #数据集预处理
    Iris_df = pd.DataFrame(Iris.data)
    Iris_df.insert(0,'target',Iris.target)

    #构造新数据集
    Iris_nolinear = Iris_df[Iris_df['target']!=0] #删除0类
    Iris_nolinear['target']-=1 # 1->0 && 2->1 
    Iris_nolinear = Iris_nolinear.sample(frac=1).reset_index(drop=True)#随机打乱
    #print(Iris_linear)

    med = MED()
    med.Holdout(Iris_nolinear, 0.7, 300)