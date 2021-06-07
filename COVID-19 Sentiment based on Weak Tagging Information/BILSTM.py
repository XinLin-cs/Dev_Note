import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from doc_vector import doc_vec

from settings import file_nolabeled, col_nolabeled, file_labeled, col_labeled

# 设置最大特征的数量，对于文本，就是处理的最大单词数量。若被设置为整数，则被限制为待处理数据集中最常见的max_features个单词
max_features=20000
# 设置每个文本序列的最大长度，当序列的长度小于maxlen时，将用0来进行填充，当序列的长度大于maxlen时，则进行截断
maxlen=50
# 设置训练的轮次
batch_size=32


def dealing_data(data_x, data_y, rate):
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=rate)
    # 查看数据大小
    print(len(train_x),'train sequences')
    print(len(test_x),'test sequences')

    print('Pad sequences (samples x time)')
    # 将文本序列处理成长度相同的序列
    train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
    print('x_train shape:', train_x.shape)
    print('x_test shape:', test_x.shape)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    return train_x,test_x,train_y,test_y


# 创建网络结构
model=Sequential()
model.add(Embedding(max_features,50,input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
# model.add(Bidirectional(LSTM(64,return_sequences=False)))

model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


# 编译模型
model.compile('adam','binary_crossentropy',metrics=['accuracy'])

# 加载弱标记数据
print("loading weak labeled data ...")
data_x,data_y = doc_vec(file_nolabeled, col_nolabeled)
train_x,test_x,train_y,test_y = dealing_data(data_x, data_y, 0.3)

# 第一次训练模型
print('1st Train...')
train_history = model.fit(train_x, train_y,batch_size=batch_size,epochs=4,validation_data=[test_x, test_y])

# 加载标记数据
print("loading labeled data ...")
data_x ,data_y = doc_vec(file_labeled, col_labeled)
train_x,test_x,train_y,test_y = dealing_data(data_x, data_y, 0.3)

# 第二次训练模型
print('2nd Train...')
train_history = model.fit(train_x, train_y,batch_size=batch_size,epochs=4,validation_data=[test_x, test_y])

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

# print(train_history.history.keys())
show_train_history(train_history,'accuracy','val_accuracy')  #查看精度变化
show_train_history(train_history,'loss','val_loss') #查看损失变化