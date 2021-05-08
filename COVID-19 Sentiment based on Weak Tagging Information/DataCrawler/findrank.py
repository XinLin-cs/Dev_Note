
from bs4 import BeautifulSoup
import re
import pandas as pd
from queue import Queue
import _thread
import time
import urllib
import string
import urllib.request

#导出路径
fileaddress = 'rank.csv'
#伪装头
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.22 Safari/537.36 SE 2.X MetaSr 1.0'}
#表情url正则式
img_pattern = re.compile(r'i_f\d\d')
#初始数据字典
total = 0
post_dict = {}
for i in range(1,70):
    name = 'i_f%02d' % (i)
    post_dict[name] = 0; 

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
        print("[ERROR] When request" , url , "WITH:", e)
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
            global total
            total+=1
            # 统计表情
            for child in commend.children:
                if child.name=='img':
                    imgid_list = img_pattern.findall(child['src'])
                    for imgid in imgid_list:
                        if post_dict[imgid] is not None:
                            post_dict[imgid]+=1
            # print(dict)
        
        # # 楼内回复内容
        # replies = soup.find_all('span',attrs={'class':'lzl_content_main'})
        # # print(replies)
        # for reply in replies:
        #     text = reply.text
        #     if text is not None:
        #         texts.append(text)
        
        # print(dataform)
    except BaseException as e:
        print("[ERROR] When request" , url)
        return None


# 线性访问
def work_by_linear(kw, start_id, terminal_id):
    try:
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

        for i in range(0,len(pidlist)):
            pid = pidlist[i]
            url = "https://tieba.baidu.com/p/%s" % (pid)
            # print(url)
            get_pdata(url)
            # print(pitem)
            # 日志显示进度
            rate = 100.0 * (i+1) / (len(pidlist))
            print("[searching posts] %.1f" % rate, "%")

    except BaseException as e:
        print("[ERROR] When searching")
        return None

if __name__ == '__main__':
    # start_id, terminal_id = 660000, 660010
    start_id, terminal_id = 1, 17
    itemlist = work_by_linear('新型冠状病毒', start_id , terminal_id )
    with open("data.txt","w") as f:
        for key in post_dict:
            if (post_dict[key]!=0):
                print('%s: %d' % (key , post_dict[key]))
                f.write('%s: %d\n' % (key , post_dict[key]))
        print('total: %d' % total)
        f.write('total: %d\n' % total)
