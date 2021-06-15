from sklearn import datasets
import pandas as pd
import numpy as np
import math


class Bayes(object):
    #初始化
    def __init__(self, koflabel):
        self.koflabel = koflabel
        self.pclass = []
        self.pnum = []
        self.ptot = []
        self.pvec = []

    # 创建贝叶斯分类器 
    def trainBayes (self, dataset, classlebels) :
        # print(self.pvec)
        num_of_sample = len (dataset)
        num_of_feature = len (dataset[0])
        for i in range(self.koflabel):
            self.pnum.append(np.ones (num_of_feature))
            self.ptot.append(num_of_feature)
            self.pclass.append(0)
        
        for i in range (num_of_sample) :
            ilabel = classlebels[i]
            self.pnum[ilabel] += dataset[i]
            self.ptot[ilabel] += sum (dataset[i])
            self.pclass[ilabel] += 1

        for i in range(self.koflabel):
            self.pclass[i]/=num_of_sample
            self.pvec.append(self.pnum[i] / self.ptot[i])

        # print(self.pvec)
        for i in range (num_of_feature):
            for j in range(self.koflabel):
                # print(i,j)
                self.pvec[j][i] = math.log (self.pvec[j][i])


    #  定义分类器 
    def classifyNB(self, vec):
        print(self.pvec)
        f , maxp = 0, 1e9
        for i in range(self.koflabel):
            p = sum(vec * self.pvec[i]) + math.log(self.pclass[i])
            if p<0:
                p =-p
            if p<maxp:
                maxp = p
                f = i
        return f


    # 验证
    def test(self, data_x, data_y):
        acc = 0
        tot = len(data_x)
        for i in range(len(data_x)):
            res = self.classifyNB(data_x[i])
            if res==data_y[i]:
                acc+=1
        accuracy = acc/tot
        print("accuracy: ", accuracy)
        return accuracy