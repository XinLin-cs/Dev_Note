import numpy as np
import math

 
#对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence,size,w2v_model):
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            sen_vec+=w2v_model.wv[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec