Pandas 索引及基本操作

```python
# 选择列

df = pd.DataFrame(np.random.rand(12).reshape(3,4)*100,
                   index = ['one','two','three'],
                   columns = ['a','b','c','d'])
print(df)

[output]:
          a            b         c         d
one    72.615321  49.816987  57.485645  84.226944
two    46.295674  34.480439  92.267989  17.111412
three  14.699591  92.754997  39.683577  93.255880

# 核心笔记：df[col]一般用于选择列，[]中写列名
# df[]默认选择列，[]中写列名（所以一般数据colunms都会单独制定，不会用默认数字列名，以免和index冲突）
data1 = df['a']
data2 = df[['a','c']]
print(data1,type(data1))
print(data2,type(data2))
# 按照列名选择列，只选择一列输出Series，选择多列输出Dataframe




#选择行
# df[]中为数字时，默认选择行，且只能进行切片的选择，不能单独选择（df[0]）
# 输出结果为Dataframe，即便只选择一行
# df[]不能通过索引标签名来选择行(df['one'])
data3 = df[:1]

# 核心笔记：df.loc[label]主要针对index选择行，同时支持指定index，及默认数字index
data3 = df.loc['one']
data4 = df.loc[['one','two']]
print(data2,type(data3))
print(data3,type(data4))
# 按照index选择行，只选择一行输出Series，选择多行输出Dataframe

df2 = pd.DataFrame(np.random.rand(16).reshape(4,4)*100,
                   columns = ['a','b','c','d'])
data2 = df2.loc[1]

data5 = df1.loc['one':'three']
data6 = df2.loc[1:3]
# 可以做切片对象
# 末端包含

# df.iloc[] - 按照整数位置（从轴的0到length-1）选择行
# 类似list的索引，其顺序就是dataframe的整数位置，从0开始计
# 和loc索引不同，不能索引超出数据行数的整数位置
df.iloc[0]
df.iloc[-1]

# 多位置索引
# 顺序可变
df.iloc[[0,2]]
df.iloc[[3,2,1]]

# 切片索引
# 末端不包含
df.iloc[1:3]
df.iloc[::2]

# 布尔型索引
df = pd.DataFrame(np.random.rand(16).reshape(4,4)*100,
                   index = ['one','two','three','four'],
                   columns = ['a','b','c','d'])
           a          b          c          d
one    74.908477  11.675143  33.440211  50.003793
two    47.673410  23.918305  72.238757  95.619127
three  30.340321  79.456831  63.338279  17.233132
four   33.549242  39.486456  48.157408  25.875854

# 不做索引则会对数据每个值进行判断
# 索引结果保留 所有数据：True返回原数据，False返回值为NaN
b1 = df < 20
df[b1]

# 单列做判断
# 索引结果保留 单列判断为True的行数据，包括其他列
b2 = df['a'] > 50
df[b2]  # 也可以书写为 df[df['a'] > 50]

# 多列做判断
# 索引结果保留 所有数据：True返回原数据，False返回值为NaN
b3 = df[['a','b']] > 50
df[b3]

# 多行做判断
# 索引结果保留 所有数据：True返回原数据，False返回值为NaN
b4 = df.loc[['one','three']] < 50
df[b4]


# 多重索引：比如同时索引行和列
# 先选择列再选择行 —— 相当于对于一个数据，先筛选字段，再选择数据量

df = pd.DataFrame(np.random.rand(16).reshape(4,4)*100,
                   index = ['one','two','three','four'],
                   columns = ['a','b','c','d'])
print(df['a'].loc[['one','three']])   # 选择a列的one，three行
print(df[['b','c','d']].iloc[::2])   # 选择b，c，d列的one，three行
print(df[df['a'] < 50].iloc[:2])   # 选择满足判断索引的前两行数据

[output]:
           a          b          c          d
one    50.660904  89.827374  51.096827   3.844736
two    70.699721  78.750014  52.988276  48.833037
three  33.653032  27.225202  24.864712  29.662736
four   21.792339  26.450939   6.122134  52.323963
------
one      50.660904
three    33.653032
Name: a, dtype: float64
               b          c          d
one    89.827374  51.096827   3.844736
three  27.225202  24.864712  29.662736
               a          b          c          d
three  33.653032  27.225202  24.864712  29.662736
four   21.792339  26.450939   6.122134  52.323963
```



在pandas中，del、drop和pop方法都可以用来删除数据，insert可以在指定位置插入数据。



```python
import pandas as pd 
from pandas import DataFrame, Series
data = DataFrame({'name':['yang', 'jian', 'yj'], 'age':[23, 34, 22], 'gender':['male', 'male', 'female']})
#data数据
'''
In[182]: data
Out[182]: 
   age  gender  name
0   23    male  yang
1   34    male  jian
2   22  female    yj
'''
#删除gender列，不改变原来的data数据，返回删除后的新表data_2。axis为1表示删除列，0表示删除行。inplace为True表示直接对原表修改。
data_2 = data.drop('gender', axis=1, inplace=False)
'''
In[184]: data_2
Out[184]: 
   age  name
0   23  yang
1   34  jian
2   22    yj
'''
#改变某一列的位置。如：先删除gender列，然后在原表data中第0列插入被删掉的列。
data.insert(0, '性别', data.pop('gender'))#pop返回删除的列，插入到第0列，并取新名为'性别'
'''
In[185]: data
Out[186]: 
       性别  age  name
0    male   23  yang
1    male   34  jian
2  female   22    yj
'''
#直接在原数据上删除列
del data['性别']
'''
In[188]: data
Out[188]: 
   age  name
0   23  yang
1   34  jian
2   22    yj
'''
```



