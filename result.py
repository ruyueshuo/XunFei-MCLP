#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/8 10:06
# @Author  : FengDa
# @File    : result.py
# @Software: PyCharm

import numpy as np
import pandas as pd

df = pd.DataFrame()
file = "result/dict.txt"
# data = np.loadtxt(file)
f = open(file)
data = f.readlines()[0][1:-1]
data = data.split(',')
for idx, d in enumerate(data):
    a = d.split(':')[0]
    a = a.strip()[1:-1]
    df.loc[idx, 'test1_file_name'] = a
    df.loc[idx, 'residual_life'] = np.float(d.split(':')[1])
df.to_csv("result/test.csv", index=False)
print('finish.')
