#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-29 11:13
# @Author  : zhangzhen
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import collections
import logging
from typing import Text, Optional, Dict, Any

import os
import time
from rasa.nlu import utils
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.nlu.components import Component

from litemind.nlu.via import next_batch
from litemind.nlu.via.n2g import get_n2g

logger = logging.getLogger(__name__)


class N2GHelper(Component):
    """A new component for n2g"""

    name = "easy_n2g"
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self,
                 component_config: Dict[Text, Any] = None):
        super(N2GHelper, self).__init__(component_config)
        self.path = component_config['data_path']
        # print(self.int2label, self.label2int)
        # init via

    def train(self, training_data, cfg, **kwargs):
        """
        统计各个词的概率
        :param training_data:
        :param cfg:
        :param kwargs:
        :return:
        """
        self.dat = []
        self.f_word_count = collections.defaultdict(int)  # 女性
        self.m_word_count = collections.defaultdict(int)  # 男性
        with codecs.open(self.path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                tokens, label = line.split(',')
                if len(tokens) == 1:
                    tokens = " " + tokens

                if len(tokens) > 1:
                    self.dat.append((tokens, label))
                    if label == '男':
                        self.m_word_count[tokens[0]] += 1
                        self.m_word_count[tokens[1]] += 1
                    else:
                        self.f_word_count[tokens[0]] += 1
                        self.f_word_count[tokens[1]] += 1

        # TEST
        acc_num = 0
        total_num = 0
        _data = []
        _label = []
        for ent in self.dat:
            name, label = ent
            w1, w2 = name[0], name[1]
            p1 = self.m_word_count[w1] / (self.m_word_count[w1] + self.f_word_count[w1]+1)
            p2 = self.m_word_count[w2] / (self.m_word_count[w2] + self.f_word_count[w2]+1)
            _data.append([p1, p2])
            _label.append(1 if label == '男' else 0)
            if w2 == ' ':
                p = p2
            else:
                p = 0.45 * p1 + 0.55 * p2
            # logging.debug("{} predict male pro {} | real gender {}".format(name, p, label))
            if p >= 0.6:
                if label == '男':
                    acc_num += 1
                else:
                    logging.warning("{} predict gender {} | real gender {}".format(name, "男", label))
                total_num += 1
            elif p <= 0.4:
                if label == '女':
                    acc_num += 1
                else:
                    logging.warning("{} predict gender {} | real gender {}".format(name, "女", label))
                total_num += 1
            else:
                logging.warning("{}, {} can't distinct.".format(name, label))

        logging.info("Use the Probability and Statistic, the Acc: {}".format(acc_num / total_num))

    def process(self, message: Message, **kwargs):
        """Process an incoming message.
            判断人物实体的性别
            1. 提取人物实体
            2. 判读实体性别
            男: 1, 女: 0, 都可能: -1
        """
        entities = message.get("spans", [])
        for ent in entities:
            # TODO 需要优化, 根据新增实体规则, 相应扩展
            if ent['label'] == 'PER':
                # 处理去掉姓氏后的名称
                per_text = message.text[ent['start'] + 1: ent['end']]
                per_len = len(per_text)

                if per_len >= 1:
                    name = " " + per_text if per_len == 1 else per_text
                    w1, w2 = name[0], name[1]
                    p1 = self.m_word_count[w1] / (self.m_word_count[w1] + self.f_word_count[w1]+1)
                    p2 = self.m_word_count[w2] / (self.m_word_count[w2] + self.f_word_count[w2]+1)
                    if w1 == ' ':
                        p = p2
                    else:
                        p = 0.45 * p1 + 0.55 * p2

                    if p >= 0.6:
                        ent.update({'gender': '男'})
                    elif p <= 0.4:
                        ent.update({'gender': '女'})
                    else:
                        ent.update({'gender': '未'})
                else:
                    ent.update({'gender': '未'})

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        # total object
        file_name = file_name + ".pkl"
        classifier_file = os.path.join(model_dir, file_name)
        utils.pycloud_pickle(classifier_file, self)
        return {"file": file_name}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['N2GHelper'] = None,
             **kwargs) -> 'N2GHelper':
        """Load this component from file."""

        file_name = meta.get("file")
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)
