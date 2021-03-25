from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import math

class MED(object):
    #初始化
    def __init__(self):
        self.vlen = 0 
        self.cnt = 0 #统计向量个数
        self.center_coordinates = {} #计算向量之和并最后求均值

    #向量距离
    def __distance(self, x, y):
        #print(x)
        #print(y)
        tot = 0
        for i in range(0,self.vlen):
            tot += (x[i]-y[i])*(x[i]-y[i])
        return math.sqrt(tot)

    #训练
    def fit(self, X_train, Y_train):
        self.vlen = len(X_train.columns)
        self.cnt = len(X_train.index)
        #计算向量之和
        for i in X_train.index:
            x = X_train.iloc[i]
            y = Y_train.iloc[i]
            if y not in self.center_coordinates.keys():
                self.center_coordinates[y]=x
            else:
                self.center_coordinates[y]=self.center_coordinates[y].add(x,fill_value=0)
        #计算均值
        for i in self.center_coordinates:
            self.center_coordinates[i]=self.center_coordinates[i]/self.cnt
    
    #寻找最近聚类
    def __closedCenter(self, x):
        mindis = 1e18
        for i in self.center_coordinates:
            if self.__distance(x,self.center_coordinates[i]) < mindis:
                mindis = self.__distance(x,self.center_coordinates[i])
                res=i
        return res

    #预测
    def predict(self, X_test):
        Y_predict = []
        for i in X_test.index:
            x = X_test.iloc[i]
            Y_predict.append(self.__closedCenter(x))
        self.Y_predict = pd.DataFrame(Y_predict)
        return self.Y_predict

    #评估
    def evaluate(self, Y_test):
        TP , FP , TN, FN = 0, 0, 0, 0
        for i in Y_test.index:
            if self.Y_predict[0][i]!=0: 
                if Y_test[i]!=0:
                    TP+=1
                else:
                    FP+=1
            else:
                if Y_test[i]!=0:
                    FN+=1
                else:
                    TN+=1
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1_Score = 2*Recall*Precision
        print("Accuracy: %f" % Accuracy)
        print("Precision: %f" % Precision)
        print("Recall: %f" % Recall)
        print("F1_Score: %f" % F1_Score)
        
        plt.scatter(X_test, Y_test, c='red', label='test')
        plt.show()
    
if __name__ == '__main__':
    #加载数据集，是一个字典类似Java中的map
    Iris = datasets.load_iris()
    #数据集预处理
    Iris_df = pd.DataFrame(Iris.data)
    Iris_df.insert(0,'target',Iris.target)

    #构造新数据集
    Iris_linear = Iris_df[Iris_df['target']!=2]#删除2类
    Iris_linear = Iris_linear.sample(frac=1).reset_index(drop=True)#随机打乱
    #print(Iris_linear)

    #生成训练集
    Iris_linear_train = pd.DataFrame(Iris_linear.iloc[0:int(len(Iris_linear)*0.7)])
    #print(Iris_linear_train)
    X_train = Iris_linear_train[[0, 1, 2]]
    Y_train = Iris_linear_train['target']
    #print(X_train)
    #print(Y_train)

    #生成测试集
    Iris_linear_test = Iris_linear.iloc[int(len(Iris_linear)*0.7):len(Iris_linear)].reset_index(drop=True)
    #print(Iris_linear_test)
    X_test = Iris_linear_test[[0, 1, 2]]
    Y_test = Iris_linear_test['target']
    #print(X_test)
    #print(Y_test)


    med = MED()
    med.fit(X_train, Y_train)
    med.predict(X_test)
    #print(med.Y_predict)
    med.evaluate(Y_test)