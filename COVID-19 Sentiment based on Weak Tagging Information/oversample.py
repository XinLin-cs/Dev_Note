from settings import file_nolabeled, col_nolabeled, file_oversample
import pandas as pd
from pandas import DataFrame

data = pd.read_csv(file_nolabeled)
data = data.dropna()
data_p = data
deta_p = data[data[col_nolabeled]>3]
data_n = data[data[col_nolabeled]<3]

print(len(data_p), len(data_n))

k = int(len(data_p)/len(data_n))-1
print(k)

data_nn = data_n.copy()
for i in range(k-1):
    data_nn = pd.concat([data_nn,data_n],axis=0)

data_n = data_nn
print(len(data_p), len(data_n))

data_res = DataFrame(pd.concat([data_p,data_n],axis=0))
data_res.to_csv(file_oversample,index=0)