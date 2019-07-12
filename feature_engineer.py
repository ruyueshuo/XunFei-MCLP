#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/11 14:23
# @Author  : FengDa
# @File    : feature_engineer.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile

from test import load_data


class feature(object):

    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        pass

    def feature_select(self):
        feature_select = SelectPercentile(chi2, percentile=95)
        feature_select.fit(self.train_x, self.train_y)
        train_csr = feature_select.transform(self.train_x)
        predict_csr = feature_select.transform(self.test_x)
        return train_csr, predict_csr

    def data_clearn(self):
        """
        去除特征值小于0的行。
        :return:
        """
        anomaly_index = np.where(self.train_x < 0)
        self.train_x = np.delete(self.train_x, anomaly_index[0], axis=0)
        self.train_y = np.delete(self.train_y, anomaly_index[0], axis=0)
        # anomaly_data = train_data[a[0]]
        # return self.train_data_new, self.train_label_new
        return self.train_x, self.train_y

    def corr_analysis(self):
        df = pd.DataFrame(data=self.train_x)
        df['label'] = self.train_y
        corr = df.corr()
        print(corr)
        pass


if __name__ == "__main__":

    train_data, train_label, train_weight, test_data, test_label, test_file = load_data()
    print(train_data.shape)
    # # df = pd.DataFrame(data=train_data)
    # # print(df.describe())
    # a = np.where(train_data<0)
    # anomaly_data = train_data[a[0]]
    # print(anomaly_data.shape)
    # print(train_data[a[0]])
    #
    # train_data_new = np.delete(train_data, a[0], axis=0)
    # print(train_data_new.shape)
    # train_label_new = np.delete(train_label, a[0], axis=0)
    # print(len(train_label_new))
    # b = np.where(train_data_new < 0)
    feat = Feature(train_x=train_data, train_y=train_label, test_x=test_data)
    feat.data_clearn()

    # train_csr, predict_csr = feat.feature_select()
    feat.corr_analysis()
    # print(train_csr.shape)