import jieba

def cut_sentence_cn(document):
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    return result