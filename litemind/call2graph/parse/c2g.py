#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-11 12:49
# @Author  : zhangzhen
# @Site    : 
# @File    : c2g.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import sre_constants
from typing import Text, Optional, Any, Dict, List, Set

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
import numpy as np
import collections
import Levenshtein

from litemind.call2graph.utils import invalid

logger = logging.getLogger(__name__)


class CallParse(Component):
    name = "call_parse"

    provides = []

    requires = []

    defaults = {}

    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None,
                 regex_titles: Optional[Dict[Text, Any]] = None,
                 regex_contents: Optional[Dict[Text, Any]] = None,
                 formula: Optional[Dict[Text, Any]] = None,
                 template_scheme: Optional[Dict[Text, Any]] = None) -> None:
        super(CallParse, self).__init__(component_config)
        self.template_scheme = template_scheme if template_scheme else {}  # # 输出映射
        self.formula = formula if formula else {}  # key: value
        self.regex_titles = regex_titles if regex_titles else collections.defaultdict(list)  # 文件头规则
        self.regex_contents = regex_contents if regex_contents else collections.defaultdict(list)  # 文件内容规则

    def add_template_scheme(self, training_data: TrainingData):
        """
        输出scheme
        :param training_data:
        :return:
        """
        self.template_scheme = training_data.entity_synonyms
        self.formula = {value: key for key, value in training_data.entity_synonyms.items()}

    def validate_regex(self, reg):
        try:
            re.compile(reg, re.I)
            return True
        except sre_constants.error as e:
            logger.warning("{} illegal regular, error msg:{}".format(reg, e.msg))
        return False

    def add_regex_title(self, training_data: TrainingData):
        """
        title 规则, 解析 intent_examples
        :param training_data:
        :return:
        """

        for example in training_data.intent_examples:
            if 'intent' in example.data:
                if self.validate_regex(example.text):
                    self.regex_titles[example.data['intent']].append(example.text)

    def add_regex_content(self, training_data: TrainingData):
        """
        context 规则, 解析 regex_features
        :param training_data:
        :return:
        """
        for feature in training_data.regex_features:
            if 'name' in feature:
                if self.validate_regex(feature['pattern']):
                    self.regex_contents[feature['name']].append(feature['pattern'])

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        self.add_template_scheme(training_data)
        self.add_regex_title(training_data)
        self.add_regex_content(training_data)

    def containsRule(self, text, rules):
        """
        判断text 是否包含rules规则
        :param text:
        :param rules:
        :return:
        """
        for rule in rules:
            pattern = re.compile(rule, re.I)  # 忽略大小写匹配
            matches = pattern.search(text)
            if matches:
                return True
        return False

    def extract_titles(self, message: Message):
        """
        处理数据的 title 映射
        :param titles:
        :return:
        """
        if not message.text:
            return {}
        tmp_titles = message.text[0]
        _mappers = collections.defaultdict(set)
        for key, vals in self.regex_titles.items():
            for i, _t in enumerate(tmp_titles):
                if invalid(_t):
                    continue
                # if key == 'thkssj':
                #     v = vals[0]
                #     p = re.compile(v)
                #     print(key, v, _t, p.search(_t))
                if self.containsRule(_t, vals):
                    _mappers[i].add(key)
                    # print(key, _t)
        return _mappers

    def extract_contents(self, data):
        if data is None:
            return {}
        _mappers = collections.defaultdict(set)
        for c in range(len(data[0])):
            col_contents = [d[c] for d in data]
            for key, vals in self.regex_contents.items():
                for i, content in enumerate(col_contents):
                    if invalid(content):
                        continue
                    if self.containsRule(content, vals):
                        _mappers[c].add(key)
        return _mappers

    def filter(self, title: Text, keys: Set[Text]):
        """
        TODO: 需要进一步优化
        处理 title 对应多个实体情况, 基于编辑距离算法选取最相似的实体
        :param title:
        :param keys:
        :return:
        """
        _keys = list(keys)
        distances = [Levenshtein.distance(title, self.formula[key]) for key in _keys]
        id = np.argmin(distances)
        return {_keys[id]}

    def order(self, titles: List[Text], key: Text, values: Set[Text]):
        """
        对处理结果进行过滤
        :param titles:
        :param key:
        :param values:
        :return:
        """

        _values = list(values)
        distances = [Levenshtein.distance(self.formula[key], titles[val]) for val in _values]
        id = np.argmin(distances)
        return [_values[id]]

    def process(self, message: Message, **kwargs: Any):
        # print(self.regex_contents)
        # print(self.regex_titles)
        # print(self.formula)
        # 判断文件头行
        titles = []

        # 验证第一行是否包含head,并基于title规则映射字段
        title_mappers = self.extract_titles(message)
        if len(title_mappers) > 0:
            titles = message.text[0]
            _data = message.text[1:]
        else:
            _data = message.text
        message.set("origin_titles", titles)
        # 基于内容映射字段
        content_mappers = self.extract_contents(_data)

        # 基于识别后字段做映射融合
        # print(titles)
        # print(title_mappers)
        # print(content_mappers)
        formula = collections.defaultdict(set)
        if titles:
            ## 融合操作
            for i, title in enumerate(titles):
                # print(content_mappers[i], content_mappers[i])
                if title_mappers[i] and content_mappers[i]:
                    merge = title_mappers[i] & content_mappers[i]
                elif title_mappers[i]:
                    merge = title_mappers[i]
                elif content_mappers[i]:
                    merge = content_mappers[i]
                else:
                    continue
                # 对于多merge基于相似过滤
                if len(merge) > 1:
                    merge = self.filter(title, merge)
                print(i, title, merge)
                formula[merge.pop()].add(i)

            formula = {key: list(indices) if len(indices) == 1 else self.order(titles, key, indices)
                       for key, indices in formula.items()}
            print(formula)

        else:
            # 不需要融合
            pass
        entities = self.add_output_scheme(formula)

        message.set("entities", entities, add_to_output=True)

    def add_output_scheme(self, formula: Dict[Text, Any]):
        """
        :return:
        """
        return {key: formula[val] if val in formula else [] for key, val in self.template_scheme.items()}

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        file_name = file_name + ".pkl"
        classifier_file = os.path.join(model_dir, file_name)
        utils.pycloud_pickle(classifier_file, self)
        return {"file": file_name}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional = None,
             **kwargs):

        file_name = meta.get("file")
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)
