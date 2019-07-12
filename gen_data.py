import os
import numpy as np
import pandas as pd

# def gen_data(path, k, dir):
#     files = os.listdir(path)
#     s = []
#     weight_list = None
#     label_list = None
#     instance_list = []
#     file_list = []
#     target_list = []
#     print(len(files))
#     ss = 0
#     for file in files:
#         if not os.path.isdir(file):
#             print(file)
#             target_list = []
#             f = open(path + "/" + file, encoding='UTF-8')
#             iter_f = iter(f)
#             str = ""
#             idx = 0
#             target = -1
#             pre_target = -1
#             weight = []
#             label = []
#             num = 0
#             for line in iter_f:
#                 if idx == 1:
#                     inst = line.split(',')
#                     inst = inst[0:-1]
#
#                     target = float(inst[0])
#                     pre_target = target
#
#                 elif idx > 1:
#                     inst = line.split(',')
#                     inst = inst[0:-1]
#                     instance = []
#                     for feature in inst:
#                         instance.append(float(feature))
#
#                     target = instance[0]
#                     if target >= 0.0 and target != pre_target:
#                         weight.append(target + 1.0)
#                         label.append(1.0)
#                         instance_list.append(instance)
#                         file_list.append(file)
#                         num += 1
#                         pre_target = target
#                         target_list.append(target)
#                 idx += 1
#             if num == 1:
#                 inst = line.split(',')
#                 inst = inst[0:-1]
#                 instance = []
#                 for feature in inst:
#                     instance.append(float(feature))
#
#                 target = instance[0]
#                 if target >= 0.0:
#                     weight.append(target + 1.0)
#                     label.append(1.0)
#                     instance_list.append(instance)
#                     file_list.append(file)
#                     target_list.append(target)
#
#             weight = np.array(weight)
#             label = np.array(label)
#             if target == 0.0:
#                 print(file)
#
#             # weight /= (target + 1.0)
#             # label *= target
#             weight /= (target + 1.0)
#             label = [target - t for t in target_list]
#
#             if weight_list is None:
#                 weight_list = weight
#             else:
#                 weight_list = np.concatenate((weight_list, weight), axis=0)
#             if label_list is None:
#                 label_list = label
#             else:
#                 label_list = np.concatenate((label_list, label), axis=0)
#
#     instance_list = np.array(instance_list)
#
#     np.save(dir + "/instance.npy", instance_list)
#     np.save(dir + "/target.npy", label_list)
#     np.save(dir + "/weight.npy", weight_list)
#     np.save(dir + "/file.npy", file_list)


def get_data(path, dir):
    files = os.listdir(path)
    s = []
    weight_list = None
    label_list = None
    instance_list = []
    file_list = []
    target_list = []
    print(len(files))
    ss = 0
    for file in files:
        if not os.path.isdir(file):
            # todo: 把设备类型作为一个特征加上，labelencoding
            print(file)
            target_list = []
            f = open(path + "/" + file, encoding='UTF-8')
            data = pd.read_csv(f)
            data = data.loc[data['部件工作时长'] >= 0]
            groupby_data = data.groupby(data['部件工作时长']).mean()
            label = groupby_data.index.tolist()
            weight = [(l + 1.0)/(label[-1] + 1) for l in label]
            for idx in groupby_data.index.tolist():
                instance_list.append(groupby_data.loc[idx].values)

            weight = np.array(weight)
            label = [label[-1] - l for l in label]
            label = np.array(label)

            file_name = []
            for i in range(len(label)):
                file_name.append(file)

            if weight_list is None:
                weight_list = weight
            else:
                weight_list = np.concatenate((weight_list, weight), axis=0)
            if label_list is None:
                label_list = label
            else:
                label_list = np.concatenate((label_list, label), axis=0)
            if file_list is None:
                file_list = file_name
            else:
                file_list = np.concatenate((file_list, file_name), axis=0)

    instance_list = np.array(instance_list)

    np.save(dir + "/instance.npy", instance_list)
    np.save(dir + "/target.npy", label_list)
    np.save(dir + "/weight.npy", weight_list)
    np.save(dir + "/file.npy", file_list)


if __name__ == '__main__':
    # get_data("train", "result/train")
    get_data("test", "result/test")
    # gen_data("train", 200, "result/train")
    # gen_data("test", 200, "result/test")
    target = np.load("train/target.npy")
    print(target)
    weight = np.load("train/weight.npy")
    print(weight)
    a = np.where(target<=0)
    print(a)
    weight = np.load("test1/weight.npy")
    print(weight)
