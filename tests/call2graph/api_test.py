#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 15:36
# @Author  : zhangzhen
# @Site    : 
# @File    : api_test.py
# @Software: PyCharm
import unittest
from datetime import datetime, timedelta
import pandas as pd
import pytest
import requests
import os

test_data = [
    ('18518067686的话单.csv', [['号码', '时间', '第三列', '类型', '时长'], ['15522453452', '2018-08-21 12:32:21', '', '短信', '1分42秒'],
                            ['15522453452', '2018-08-21 12:32:21', '', '短信', '1分42秒']]),
]


@pytest.mark.parametrize("filename, data", test_data)
def test_call_graph_url(filename, data):
    url = 'http://127.0.0.1:5000/call2graph'
    postData = {
        "project": "c2g",
        "filename": filename,
        "data": data
    }
    res = requests.post(url, json=postData)
    print(res.text)


test_files = [
    # ("tests/data/call2graph", "2011年10月电信18910744589通话详单(1).xls"),
    # ("tests/data/call2graph", "2011年10月电信18910744589通话详单.xls"),
    ("tests/data/call2graph", "本机与对方都有的移动标准话单.xlsx"),
    # ("tests/data/call2graph", "本机与对方都有的移动标准话单 - 副本.xlsx"),
    # ("tests/data/call2graph", "话单数据.xlsx"),
]


@pytest.mark.parametrize("file_dir, filename", test_files)
def test_read_file(file_dir, filename):
    # get excel data
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
    # post api
    url = 'http://127.0.0.1:5000/call2graph'
    postData = {
        "project": "c2g",
        "filename": filename,
        "data": data
    }
    print(postData)
    res = requests.post(url, json=postData)
    print(res.text)


testdata = [
    (datetime(2001, 12, 12), datetime(2001, 12, 11), timedelta(1)),
    (datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1)),
]


@pytest.mark.parametrize("a,b,expected", testdata)
def test_timedistance_v0(a, b, expected):
    diff = a - b
    assert diff == expected


@pytest.mark.parametrize("a,b,expected", testdata, ids=["forward", "backward"])
def test_timedistance_v1(a, b, expected):
    diff = a - b
    assert diff == expected


if __name__ == '__main__':
    unittest.main()
