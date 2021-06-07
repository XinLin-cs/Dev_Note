import jieba
import re

pattern = re.compile(r'[]，。！“”：,.!~@#*？?&;":(-+]()')

def cut_sentence_cn(document):
    return document
    # 删除特殊符号
    document = re.sub(pattern,'',document)
    
    # 分词
    document_cut = jieba.cut(document)
    stopwords = stopwordslist('./WordVector/stopword.txt') 
    outstr = '' #设置一个空的字符串，用于储存结巴分词后的句子
    for word in document_cut: #遍历分词后的每一个单词
        if word not in stopwords: #如果这个单词不在停用表里面
            if word != '\t': #且这个单词不是制表符
                outstr += word #就将这个词添加到结果中
                outstr += " " #但是这里为什么又要添加双引号中间带空格？
                              #测试了一下，原来是为了让结巴分词后的词以空格间隔分割开
    if len(outstr)<=5:
        return ""
    return outstr
    

 
# 创建停用词list函数
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()] #分别读取停用词表里的每一个词，
                                                                                               #因为停用词表里的布局是一个词一行
    return stopwords #返回一个列表，里面的元素是一个个的停用词