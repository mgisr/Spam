#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/22 20:24
# @Author  : mgisr
# @Site    : Sdust ShanDong China
# @File    : load_data.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path, train_size=0.8):
    """
    从指定路径加载垃圾邮件数据
    :param path: 数据集路径
    :param train_size: 训练集占数据集的比重，默认为0.8
    :return: 按比例分割后的数据集
    """

    class_names = ['labels', 'messages']
    data = pd.read_csv(path, sep='\\t', header=None, names=class_names, engine='python')
    datas, labels = data['messages'], data['labels']
    _train_data, _test_data, _train_label, _test_label = train_test_split(datas, labels, train_size=train_size,
                                                                       random_state=520)

    return np.array(_train_data), np.array(_test_data), np.array(_train_label), np.array(_test_label)

