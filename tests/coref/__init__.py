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
import pytest
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.training_data import Message

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
    data = None
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