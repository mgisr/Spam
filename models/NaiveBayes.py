#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/22 20:50
# @Author  : mgisr
# @Site    : Sdust ShanDong China
# @File    : NaiveBayes.py
# @Software: PyCharm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


def transform(x, model='ft'):
    """
    词频向量化
    :param x: 文本
    :param model: 向量化模式，值为ft表示使用fit_transform，为t表示使用transform
    :return: 词频矩阵
    """

    if model is 'ft':
        return CountVectorizer().fit_transform(x)
    else:
        return CountVectorizer().transform(x)


class NaiveBayes:

    def __init__(self):
        self.x = None
        self.y = None
        self.model = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.model = BernoulliNB().fit(x, y)

    def predict(self, text):
        """
        将文本列表中的文本进行分类
        :param text: 文本列表
        :return: 分类后的结果
        """

        return self.model.predict(transform(text, 't'))
