#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-01 10:30
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
"""
    数据获取、处理的工具集
"""
import random


def next_batch(data, labels, batch_size=128, shuffle=True):
    """
    生成batch数据, 用于训练模型
    :param data:
    :param labels:
    :param batch_size:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    for i in range(int((len(data) + 1) / batch_size)):
        if (i + 1) * batch_size > len(data):
            yield (data[i * batch_size:], labels[i * batch_size:])
        else:
            yield (data[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])