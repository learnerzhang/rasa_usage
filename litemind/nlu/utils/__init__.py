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

import os

import requests
from rasa.nlu.tokenizers import Token

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HttpSessionContext:
    """ 根据requests session 创建 http请求的上下文环境, 用于with 或者手动创建连接, 关闭连接

    """

    def __init__(self):
        self.session = requests.Session()

    def __enter__(self):
        self._set_adapter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return exc_type is not None

    def _set_adapter(self):
        a = requests.adapters.HTTPAdapter(max_retries=3)
        b = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount('http://', a)
        self.session.mount('https://', b)

    def open(self):
        self._set_adapter()
        return self.session

    def close(self):
        self.session.close()


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


def duplicate(entities):
    # type:(List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
    if entities:
        d = {ent['entity'] + "_" + str(ent['start']) + "_" + str(ent['end']): ent for ent in entities}
        return [val for val in d.values()]
    return []


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


def tokenizer_extract(token):
    """
    rasa token 封装
    :param token:
    :return:
    """
    if isinstance(token, Token):
        return token.text
    return token



