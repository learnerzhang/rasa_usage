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
from typing import Text, Any, List, Dict


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


def legalPronouns(pronouns: Text):
    """
    :param pronouns:
    :return:
    """
    # TODO
    ps = ['他', '他们', '他俩', '她', '她们', '她俩', '它', '它们', '该片', '此楼', '该厂']
    if pronouns in ps:
        return True
    else:
        return False


def isSingular(pronouns: Text):
    """
    判断代词单复数  singular or plural
    :param pronouns:
    :return: 1: 单数, 0:复数, -1:未知
    """

    # TODO
    singulars = ['它', '他', '她', '其']
    plural = ['他们', '它们', '她们']
    if pronouns in singulars:
        return 1
    elif pronouns in plural:
        return 0
    else:
        return -1


def pronouns2gender(pronouns: Text):
    """
    判断代词的性别
    :param word:
    :return:
    """
    # TODO
    genders = {
        "男": ['他', '他们', '他俩'],
        "女": ['她', '她们', '她俩'],
        "它": ['它', '它们', '该片', '此楼', '该厂'],
    }
    for key, vals in genders.items():
        if pronouns in vals:
            return key
    else:
        return "中"


def span_output_format(spans: List[Dict]):
    """
    移除不必要的属性
    :param spans:
    :return:
    """
    for span in spans:
        if 'gender' in span:
            del span['gender']
