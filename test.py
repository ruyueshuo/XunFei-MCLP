import numpy as np
import pandas as pd
from gcForest import *
from feature_engineer import *
from lgbModel import lgbModel
import time


def load_data():
    train_data = np.load('result/train/instance.npy', allow_pickle=True)
    train_label = np.load('result/train/target.npy')
    train_weight = np.load('result/train/weight.npy')
    test_data = np.load('result/test/instance.npy')
    test_label = np.load('result/test/target.npy')
    test_file = np.load('result/test/file.npy')
    return [train_data, train_label, train_weight, test_data, test_label, test_file]


def sample_data(data_list, fraction=1):
    length = data_list[0].shape[0]
    index = np.arange(np.int(length / fraction)) * fraction
    data_list[0] = data_list[0][index, :]
    data_list[1] = data_list[1][index]
    data_list[2] = data_list[2][index]
    # data_list[3] = data_list[3][index, :]
    # data_list[4] = data_list[4][index]
    # data_list[5] = data_list[5][index]
    return data_list


if __name__ == '__main__':
    train_data, train_label, train_weight, test_data, test_label, test_file = load_data()
    # test
    # train_label = np.insert(train_label, train_label[-1])
    data_list = [train_data, train_label, train_weight, test_data, test_label, test_file]
    train_data, train_label, train_weight, test_data, test_label, test_file = sample_data(data_list, fraction=1)
    a = np.where(train_label<0)
    feat = feature(train_x=train_data, train_y=train_label, test_x=test_data)
    train_data, train_label = feat.data_clearn()


    clf = lgbModel()
    m = clf.built_model(learning_rate=0.5)
    prediction = clf.train(m, train_data, train_label, test_data, n_splits=5)

    result = {}
    files = list(set(test_file))
    for i, file in enumerate(files):
        res = 0
        index = np.where(test_file == file)
        num = len(index[0])
        for idx in index[0]:
            res = prediction[idx] + test_label[idx]
            if file in result:
                result[file] = (result[file] + res) / 2
            else:
                result[file] = res
        result[file] -= test_label[index[0][-1]]
        # residual_life = res / num - prediction[index[0][-1]]
        if result[file] < 0:
            result[file] = 0
        # result[file] = residual_life

    print(result)

    # print(a)
    # clf = gcForest(num_estimator=100, num_forests=2, max_layer=1, max_depth=100, n_fold=5)
    # start = time.time()
    # # train_weight = np.ones((train_data.shape[0]))
    # clf.train(train_data, train_label, train_weight)
    # # clf.train(train_data, train_label, train_weight)
    # end = time.time()
    # print("fitting time: " + str(end - start) + " sec")
    # start = time.time()
    # prediction = clf.predict(test_data)
    # end = time.time()
    # print("prediction time: " + str(end - start) + " sec")


    # result = {}
    # for index, item in enumerate(test_file):
    #     if item not in result:
    #         result[item] = prediction[index]
    #     else:
    #         result[item] = (result[item] + prediction[index]) / 2
    # print(result)
    df = pd.DataFrame()
    for idx, d in enumerate(result):
        df.loc[idx, 'test1_file_name'] = d
        df.loc[idx, 'residual_life'] = result[d]
    now = time.time()
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    df.to_csv("result/result_{}.csv".format(otherStyleTime), index=False)
