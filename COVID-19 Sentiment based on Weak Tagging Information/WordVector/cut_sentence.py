import jieba
import re

pattern = re.compile(r'[]，。！“”：,.!~@#*？?&;\":(-+]()')

def cut_sentence_cn(document):
    # 删除特殊符号
    document = re.sub(pattern,'',document)
    # 分词
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    return result