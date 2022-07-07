import pandas as pd
import glob
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

files = glob.glob("data/*")
datas = []

for f in files:
    data = pd.read_excel(f, sheet_name='file')
    data = data.dropna(axis=0, how='any')
    datas.append(data)


def get_data(A):
    data = []
    for item in ["日期", "收盘价(元)"]:
        data.append(list(A[item]))
    dic_data = {}
    for i in range(len(datas)):
        dic_data[str(data[0][i])] = data[1][i]
    return dic_data


def pearson_score(A, B):
    data_A = get_data(A)
    data_B = get_data(B)
    same_time = set(list(data_A.keys())) & set(list(data_B.keys()))
    num_A = [data_A[key] for key in same_time]
    num_B = [data_B[key] for key in same_time]
    r = pearsonr(num_A, num_B)
    return r


LEN_DATA = len(datas)
person_matrix = np.zeros((LEN_DATA, LEN_DATA))
for i in range(LEN_DATA):
    for j in range(LEN_DATA):
        try:
            person_matrix[i, j] = pearson_score(datas[i], datas[j])[0]
        except:
            person_matrix[i, j] = 0

plt.matshow(person_matrix)
plt.show()

print(person_matrix)
