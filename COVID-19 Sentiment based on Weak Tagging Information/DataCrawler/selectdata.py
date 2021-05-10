import pandas as pd

#导入配置
index_label = 'id'
fileaddress = 'data.csv'
spliter = ','

dataform = pd.read_csv(fileaddress, encoding="utf_8_sig", sep=spliter, index_col='id')
data_good = pd.DataFrame(columns = dataform.columns)
data_bad = pd.DataFrame(columns = dataform.columns)

print('[selecting...]')
for r in dataform.index:
    flag = 0
    for c in dataform: 
        if c=='text' or c=='id':
            continue
        if int(dataform[c][r])>0:
            flag = 1
            break
    if flag == 1:
        # print(r)
        data_good = data_good.append(dataform.iloc[r])
    else:
        data_bad = data_bad.append(dataform.iloc[r])

print('[saving...]')
data_good.to_csv('data_good.csv', index_label = index_label , encoding="utf_8_sig", sep=spliter)
data_bad.to_csv('data_bad.csv', index_label = index_label , encoding="utf_8_sig", sep=spliter)

print('[select finish]')