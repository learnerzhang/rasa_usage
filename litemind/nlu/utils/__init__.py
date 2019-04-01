#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-01 10:16
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
"""
    工具: 词性、句法分析
"""
from typing import Text, Any, List


def get_start(idx: int, tokens: List):
    """
    词组索引->重定向句子中索引
    :param idx:
    :param tokens:
    :return:
    """
    start = 0
    for i, w in enumerate(tokens):
        if i == idx:
            break
        start += len(w)
    return start
