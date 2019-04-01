#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-29 16:05
# @Author  : zhangzhen
# @Site    : 
# @File    : n2g.py
# @Software: PyCharm
import codecs
import collections
import random

import numpy as np
from typing import Text, Dict


def read_data(path: Text, vocab2int: Dict):
    """
    读取姓名文件,
    :param path:
    :param vocab2int:
    :return:
    """
    dat = []
    labels = []
    with codecs.open(path, encoding='utf-8') as f:
        for line in f.readlines():
            words, tag = line.strip().split(",")
            for w in words:
                if w not in vocab2int:
                    vocab2int[w] = len(vocab2int) + 1
            dat.append([w for w in words])
            labels.append(tag)
    return dat, labels


def words2int(words, vocab2int, max_length=8):
    """
    字符转换成idx
    :param words:
    :param vocab2int:
    :param max_length:
    :return:
    """
    return [[vocab2int[w] for w in word] + [0] * (max_length - len(word)) for word in words]


def get_n2g(filepath, label2int, max_length):
    """
    获取(姓名,性别字段)
    :return:
    """
    vocabulary2int = collections.defaultdict(int)  # word -> idx

    dat, labels = read_data(filepath, vocabulary2int)

    # input-x
    dat = words2int(dat, vocabulary2int, max_length=max_length)
    # input-y
    labels = [label2int[label] for label in labels]

    return np.array(dat), np.array(labels), vocabulary2int



