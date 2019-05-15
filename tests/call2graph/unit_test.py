#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 15:36
# @Author  : zhangzhen
# @Site    : 
# @File    : unit_test.py
# @Software: PyCharm
import unittest
import pytest
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

import os
import pandas as pd
import pprint

examples = [
    (Message([
        ['号码', '时间', '第三列', '类型', '时长'],
        ['15522453452', '2018-08-21 12:32:21', '', '短信', '1分42秒'],
        ['15522453452', '2018-08-21 12:32:21', '', '短信', '1分42秒'],
        ['15522453452', '08-21', '', '短信', '1时12分42秒'],
        ['15522453452', '08-21 12:32:21', '', '短信', '42秒'],
        ['15522453452', '12:32:21', '', '短信', '42秒'],
        ['15522453452', '08/21 12:32:21', '', '短信', '1分42秒'],
        ['15522453452', '2018/06', '', '短信', '42秒'],
    ]), '18518067686的话单.csv',),
]


@pytest.mark.parametrize("message, filename", examples)
def test_c2g_parse(message, filename):
    component_meta = {
        "name": "call_parse",
        "file": "component_0_call_parse.pkl",
        "class": "litemind.call2graph.parse.c2g.CallParse"
    }
    model_dir = 'models/c2g/model'
    call2parse = ComponentBuilder().load_component(component_meta, model_dir, Metadata({}, None))
    call2parse.process(message)


test_files = [
    # ("tests/data/call2graph", "2011年10月电信18910744589通话详单(1).xls"),
    # ("tests/data/call2graph", "2011年10月电信18910744589通话详单.xls"),
    # ("tests/data/call2graph", "本机与对方都有的移动标准话单.xlsx"),
    # ("tests/data/call2graph", "本机与对方都有的移动标准话单 - 副本.xlsx"),
    # ("tests/data/call2graph", "话单数据.xlsx"),
    ("tests/data/call2graph", "demo1.xls"),
]


@pytest.mark.parametrize("file_dir, filename", test_files)
def test_c2g_file_parse(file_dir, filename):
    path = os.path.join(file_dir, filename)
    data = []
    if os.path.isfile(path):
        dat_excel = pd.read_excel(path, header=None)
        titles = list(dat_excel.columns)
        for i, r in dat_excel.iterrows():
            row = []
            for t in titles:
                if pd.notna(r[t]):
                    row.append(str(r[t]))
                else:
                    row.append(None)
            data.append(row)

    message = Message(data)
    component_meta = {
        "name": "call_parse",
        "file": "component_0_call_parse.pkl",
        "class": "litemind.call2graph.parse.c2g.CallParse"
    }
    model_dir = 'models/c2g/model'
    call2parse = ComponentBuilder().load_component(component_meta, model_dir, Metadata({}, None))
    call2parse.process(message)
    pprint.pprint(message.data)


if __name__ == '__main__':
    unittest.main()
