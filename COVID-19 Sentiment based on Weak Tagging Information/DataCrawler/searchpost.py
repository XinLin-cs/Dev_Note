import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from queue import Queue
import _thread
import time
import urllib
import string

#导出配置
index_label = 'id'
fileaddress = 'data.csv'
spliter = ','
#伪装头
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.22 Safari/537.36 SE 2.X MetaSr 1.0'}
#表情url正则式
img_pattern = re.compile(r'i_f\d\d')
#初始数据字典
post_dict = {
    'text':0,
    'i_f25':0,
    'i_f16':0,
    'i_f05':0,
    'i_f01':0,
    'i_f02':0,
    'i_f04':0,
    'i_f15':0,
    'i_f27':0,
    'i_f28':0,
    'i_f30':0,
}

# 数据清洗
def clear_data(text):
    try:
        text = re.sub('[\n\r ]', '', text)
        text = re.sub('[,]', '，', text)
        return text
    except BaseException as e:
        # print("ERROR: When request" , url , "WITH:", e)
        return text
    

# 访问url 返回数据元组
def get_pids(url):
    try:
        url=urllib.parse.quote(url,safe=string.printable)
        res=urllib.request.urlopen(url)
        # 生成soup对象
        soup = BeautifulSoup(res, 'lxml')
        # print(soup)
        posts = soup.find_all('li',attrs={'class':'j_thread_list clearfix thread_item_box'})
        pids = []
        for it in posts:
            pids.append(it['data-tid'])
        # print(pids)
        return pids
    except BaseException as e:
        # print("ERROR: When request" , url , "WITH:", e)
        return None

def get_pdata(url):
    try:
        url=urllib.parse.quote(url,safe=string.printable)
        res=urllib.request.urlopen(url).read()
        # 生成soup对象
        soup = BeautifulSoup(res, 'lxml')
        # print(soup)
        # 获取帖子内容
        dataform = []
        # 楼中内容
        commends = soup.find_all('div',attrs={'class':'d_post_content j_d_post_content clearfix'})
        for commend in commends:
            # 选取含文本评论
            # 文本
            text = clear_data(commend.text)
            if text is None or text =='':
                continue
            # 初始化
            dict = post_dict.copy()
            dict['text'] = text
            # 统计表情
            for child in commend.children:
                if child.name=='img':
                    imgid_list = img_pattern.findall(child['src'])
                    for imgid in imgid_list:
                        if dict[imgid] is not None:
                            dict[imgid]+=1
            dataform.append(dict)
            # print(dict)
        
        # # 楼内回复内容
        # replies = soup.find_all('span',attrs={'class':'lzl_content_main'})
        # # print(replies)
        # for reply in replies:
        #     text = reply.text
        #     if text is not None:
        #         texts.append(text)
        
        # print(dataform)
        return dataform
    except BaseException as e:
        # print("ERROR: When request" , url , "WITH:", e)
        return None


# 线性访问
def work_by_linear(kw, start_id, terminal_id):
    # 中文转码
    kw_code = urllib.parse.quote(kw)
    # print(kw_code)
    pidlist = []
    # 获取吧内帖子id表
    for i in range(start_id, terminal_id):
        url = "https://tieba.baidu.com/f?kw=%s&pn=%d" % (kw_code,50*i)
        # print(url)
        pids = get_pids(url)
        for pid in pids:
            pidlist.append(pid)
        # 日志显示进度
        rate = 100.0 * (i - start_id + 1) / (terminal_id - start_id)
        print("[collecting posts] %.1f" % rate, "%")
    # print (pidlist)

    dataform = []
    for i in range(0,len(pidlist)):
        pid = pidlist[i]
        url = "https://tieba.baidu.com/p/%s" % (pid)
        # print(url)
        pdata = get_pdata(url)
        if pdata is not None:
            for pitem in pdata:
                dataform.append(pitem)
            
        # 日志显示进度
        rate = 100.0 * (i+1) / (len(pidlist))
        print("[searching posts] %.1f" % rate, "%")
    return dataform


# 参数为数据表，保存为csv文件
def save_as_csv(data):
    data_df = {}
    for it in data:
        for key in it:
            if data_df.get(key) is None:
                data_df[key] = []
            data_df[key].append(it[key])
    data_df = pd.DataFrame(data_df)
    data_df.to_csv(fileaddress, index_label = index_label , encoding="utf_8_sig", sep=spliter)


if __name__ == '__main__':
    # start_id, terminal_id = 660000, 660010
    start_id, terminal_id = 1, 30
    itemlist = work_by_linear('新型冠状病毒', start_id , terminal_id )
    # print(itemlist)
    save_as_csv(itemlist)

