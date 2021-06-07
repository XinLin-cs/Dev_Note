import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import Text8Corpus
from WordVector.cut_sentence import cut_sentence_cn

from settings import file_nolabeled, col_nolabeled, col_comment , model_address


# 构建word2vec模型，词向量的训练与生成
def get_dataset_vec(dataset):
    data_list=[]
    for i in range(len(dataset)):
        data_list.append(str(dataset[i]).split())
    
    n_dim = 50
    w2v_model = Word2Vec(data_list, sg=1, vector_size=n_dim, min_count=20, hs=0)  # 初始化模型并训练
    # w2v_model = Word2Vec(data_list, sg=1, vector_size=n_dim, min_count=0, hs=0)  # 初始化模型并训练
    w2v_model.save(model_address)  # 保存训练结果


if __name__ == '__main__':
    df = pd.read_csv(file_nolabeled)
    # print(df)
    df = df[[col_comment, col_nolabeled]]

    cw=lambda x:cut_sentence_cn(x)

    # #查看分词结果
    # df[col_comment].apply(cw).to_csv('./data/temp.csv')

    data = np.array(df[col_comment].apply(cw))

    get_dataset_vec(data)