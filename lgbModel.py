#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 19:08
# @Author  : FengDa
# @File    : lgbModel.py
# @Software: PyCharm
import numpy as np
# import pandas as pd
# import time
# import datetime
# import gc
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
# # from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold
# from sklearn.metrics import roc_auc_score, log_loss
# from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.dot(temp, temp) / len(temp)
    return 'eval_loss', res, False

# 模型部分
class lgbModel(object):

    def __init__(self):
        # self.learning_rate = learning_rate
        self.model = None
        pass

    def built_model(self, learning_rate=0.5):
        model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=128, max_depth=-1, learning_rate=learning_rate,
                                   n_estimators=2000,
                                   max_bin=425, subsample_for_bin=50000, objective='regression', min_split_gain=0,
                                   min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
        self.model = model
        return model

    def train(self, model, X_loc_train, y_loc_train, X_loc_test, n_splits=5):
        n_splits = n_splits
        seed = 1024
        skf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        baseloss = []
        loss = 0
        result = 0
        for i, (train_index, test_index) in enumerate(skf.split(X_loc_train, y_loc_train)):
            print("Fold", i)
            lgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                                  eval_names=['train', 'valid'],
                                  # fobj=compute_loss,
                                  # feval=compute_loss,
                                  eval_metric=compute_loss,
                                  # eval_metric='auc',
                                  eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                            (X_loc_train[test_index], y_loc_train[test_index])],
                                  early_stopping_rounds=100)
            # baseloss.append(lgb_model.best_score_['valid'][compute_loss])
            # loss += lgb_model.best_score_['valid'][compute_loss]
            test_pred = lgb_model.predict(X_loc_test, num_iteration=lgb_model.best_iteration_)
            print('test mean:', test_pred.mean())
            result += test_pred

        # print('logloss:', baseloss, loss / 5)
        return result / n_splits

    def compute_loss(self, target, predict):
        temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
        res = np.dot(temp, temp) / len(temp)
        return res

    def predict(self):
        pass


if __name__ == "__main__":

    print('a')
# # 加权平均
# res['predicted_score'] = 0
# for i in range(5):
#     res['predicted_score'] += res['prob_%s' % str(i)]
# res['predicted_score'] = res['predicted_score']/5
#
# # 提交结果
# mean = res['predicted_score'].mean()
# print('mean:',mean)
# now = datetime.datetime.now()
# now = now.strftime('%m-%d-%H-%M')
# res[['instance_id', 'predicted_score']].to_csv("G:/Competition/DC竞赛/2018科大讯飞AI营销算法大赛/result/lgb_baseline_%s.csv" % now, index=False)

#
# output.to_csv("G:/Competition/DC竞赛/2018科大讯飞AI营销算法大赛/result/submit_20180928-lightgbm.csv",
#               index=False, encoding="utf-8")
# pd.DataFrame({'features':train_features, 'imp':lgb_model_step.feature_importance()}).sort_values('imp',ascending=False)