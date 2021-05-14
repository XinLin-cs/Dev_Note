from typing import Text
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from WordVector.sentence_vector import build_sentence_vector
from WordVector.cut_sentence import cut_sentence_cn

file_address = 'data/JDComment_data.csv'
model_address = 'data/w2v_model_100.pkl'
result_address = 'data/vector_data.csv'

# 将文本数据转换为文本向量
def doc_vec():
    data = pd.read_csv(file_address)
    w2v_model = Word2Vec.load(model_address)   #加载训练好的Word2Vec模型
 
    #读取词权重字典
    #with open('data/key_words_importance', 'r') as f:
       #key_words_importance = eval(f.read())
    
    #数据集处理
    data_y = np.array(data['得分'].apply(lambda x:((float(x)-1)/4)))
    text_list = np.array(data['评论内容'].apply(lambda x:cut_sentence_cn(x)))

    #训练集转换为向量
    data_list=[]
    for i in range(len(text_list)):
        data_list.append(str(text_list[i]).split())
    
    data_x=np.concatenate([build_sentence_vector(sen,100,w2v_model) for sen in data_list])
    for i in np.nditer(data_x, op_flags=['readwrite']): 
        for x in np.nditer(i, op_flags=['readwrite']): 
            if x<0:
                x *= 0.5
                x += 1
    return data_x,data_y

if __name__ == '__main__':
    data_x,data_y = doc_vec()
    df = pd.concat([pd.DataFrame(data_y), pd.DataFrame(data_x)],axis=1,ignore_index=True)
    print(df)
    df.to_csv(result_address)