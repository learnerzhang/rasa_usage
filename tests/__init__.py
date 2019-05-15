#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-29 15:19
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
import unittest
import paramunittest

# content of test_time.py
import pytest
from datetime import datetime, timedelta


# 方案一
@paramunittest.parametrized(
    (1, 2, 3),
    (1, 2, 3),
    (1, 2, 3),
    (1, 2, 3),
)
class Test_Add(paramunittest.ParametrizedTestCase):

    def setParameters(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def test_add(self):
        assert self.a + self.b == self.c

        self.assertLess(self.a, self.b)


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
