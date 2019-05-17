#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 14:59
# @Author  : zhangzhen
# @Site    : 
# @File    : lite.py
# @Software: PyCharm
from typing import Any, Dict, Text

from rasa.nlu.emulators import NoEmulator


class CallEmulator(NoEmulator):
    def __init__(self) -> None:
        self.name = "call"

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return data

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""

        return data
