from keras.models import Sequential
from keras.layers import Dense,LSTM,Bidirectional
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from doc_vector import doc_vec

#载入数据
def read_data():
    mnist=doc_vec()
    train_x,train_y=mnist.train.images.reshape(-1,28,28),mnist.train.labels,
    valid_x,valid_y=mnist.validation.images.reshape(-1,28,28),mnist.validation.labels,
    test_x,test_y=mnist.test.images.reshape(-1,28,28),mnist.test.labels
    return train_x,train_y,valid_x,valid_y,test_x,test_y
 
#双向LSTM模型
def BiLSTM(train_x,train_y,valid_x,valid_y,test_x,test_y):
    #创建模型
    model=Sequential()
    lstm=LSTM(64,input_shape=(28,28),return_sequences=False)  #返回最后一个节点的输出
    model.add(Bidirectional(lstm))  #双向LSTM
    model.add(Dense(10,activation='softmax'))
    #编译模型
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #训练模型
    model.fit(train_x,train_y,batch_size=500,nb_epoch=15,verbose=2,validation_data=(valid_x,valid_y))
    #查看网络结构
    model.summary()
    #评估模型
    pre=model.evaluate(test_x,test_y,batch_size=500,verbose=2)
    print('test_loss:',pre[0],'- test_acc:',pre[1])
   
train_x,train_y,valid_x,valid_y,test_x,test_y=read_data()
BiLSTM(train_x,train_y,valid_x,valid_y,test_x,test_y)