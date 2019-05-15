#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 11:26
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
"""
    指代消解测试: 功能测试及api测试
"""
import pprint
import unittest

import pytest
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

test_texts = [
    ("李世民病倒了，小明说他是累病的"),
]


@pytest.mark.parametrize("text", test_texts)
def test_ltp(text):
    message = Message(text)
    component_meta = {
        "name": "ltp",
        "path": "/Users/zhangzhen/data/ltp_data_v3.4.0",
        "lexicon": "lexicon",
        "dimension": {
            "Nh": "PER",
            "Ni": "ORG",
            "Ns": "LOC"
        },
        "class": "litemind.nlu.utils.ltp.LtpHelper"
    }
    model_dir = 'models/coref/model_20190515-150912'
    ltp = ComponentBuilder().load_component(component_meta, model_dir, Metadata({}, None))
    ltp.process(message)
    pprint.pprint(message.data)


@pytest.mark.parametrize("text", test_texts)
def test_n2g_parse(text):
    message = Message(text)
    spans = [
        {'end': 3, 'label': 'PER', 'start': 0},
        {'end': 9, 'label': 'PER', 'start': 7},
        {'end': 11, 'label': 'Pronoun', 'start': 10}
    ]
    message.set("spans", spans, add_to_output=True)
    component_meta = {
        "name": "easy_n2g",
        "data_path": "data/n2g/name_dev.dat",
        "file": "component_1_easy_n2g.pkl",
        "class": "litemind.nlu.utils.n2g.easy_n2g.N2GHelper"
    }
    model_dir = 'models/coref/model_20190515-150912'
    n2g = ComponentBuilder().load_component(component_meta, model_dir, Metadata({}, None))
    n2g.process(message)
    pprint.pprint(message.data)


@pytest.mark.parametrize("text", test_texts)
def test_coref_parse(text):
    message = Message(text)
    model_dir = 'models/coref/model_20190515-150912'
    ltp_component_meta = {
        "name": "ltp",
        "path": "/Users/zhangzhen/data/ltp_data_v3.4.0",
        "lexicon": "lexicon",
        "dimension": {
            "Nh": "PER",
            "Ni": "ORG",
            "Ns": "LOC"
        },
        "class": "litemind.nlu.utils.ltp.LtpHelper"
    }
    ltp = ComponentBuilder().load_component(ltp_component_meta, model_dir, Metadata({}, None))
    ltp.process(message)

    spans = [
        {'end': 3, 'gender': '男', 'label': 'PER', 'start': 0},
        {'end': 9, 'gender': '男', 'label': 'PER', 'start': 7},
        {'end': 11, 'label': 'Pronoun', 'start': 10}
    ]
    message.set("spans", spans, add_to_output=True)
    component_meta = {
        "name": "stg",
        "w2v_path": "/Users/zhangzhen/data/emb_ch/embedding.50.cformat",
        "class": "litemind.coref.stg.Strategy"
    }
    coref_stg = ComponentBuilder().load_component(component_meta, model_dir, Metadata({}, None))
    coref_stg.process(message)
    pprint.pprint(message.data)


if __name__ == '__main__':
    unittest.main()
