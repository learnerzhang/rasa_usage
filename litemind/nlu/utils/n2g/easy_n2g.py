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

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "easy_n2g"
    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = []

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
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
            p1 = self.m_word_count[w1] / (self.m_word_count[w1] + self.f_word_count[w1])
            p2 = self.m_word_count[w2] / (self.m_word_count[w2] + self.f_word_count[w2])
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
        entities = message.get("entities", [])
        for ent in entities:
            # TODO 需要优化, 根据新增实体规则, 相应扩展
            if ent['dim'] == 'Nh':
                """如果是人实体, 则..."""
                if len(ent['value']) > 1:
                    name = " " + ent['value'] if len(ent['value']) == 1 else ent['value']
                    w1, w2 = name[0], name[1]
                    p1 = self.m_word_count[w1] / (self.m_word_count[w1] + self.f_word_count[w1])
                    p2 = self.m_word_count[w2] / (self.m_word_count[w2] + self.f_word_count[w2])
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
