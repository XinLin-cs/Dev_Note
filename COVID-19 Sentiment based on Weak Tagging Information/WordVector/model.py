import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from cut_sentence import cut_sentence_cn


file_address = 'data/JDComment_data.csv'
model_address = 'data/w2v_model_300.pkl'


# 构建word2vec模型，词向量的训练与生成
def get_dataset_vec(dataset):
    n_dim = 300
    w2v_model = Word2Vec(dataset, sg=1, vector_size=n_dim, min_count=10, hs=0)  # 初始化模型并训练
    w2v_model.save(model_address)  # 保存训练结果


if __name__ == '__main__':
    df = pd.read_csv(file_address)
    df = df[['评论内容','得分']]

    cw=lambda x:cut_sentence_cn(x)
    data = np.array(df['评论内容'].apply(cw))
    
    get_dataset_vec(data)