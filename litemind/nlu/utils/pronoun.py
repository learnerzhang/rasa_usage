#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-01 11:51
# @Author  : zhangzhen
# @Site    : 
# @File    : pronoun.py
# @Software: PyCharm
from typing import Text, Optional, Any, Dict

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

from litemind.nlu.utils import get_start


class PronounHelper(Component):
    """A new component"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "pronoun"

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

    def __init__(self, component_config: Dict[Text, Any] = None):
        """
        代词处理逻辑
        """
        super(PronounHelper, self).__init__(component_config)

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training via. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        pass

    def legalPronouns(self, pronouns: Text):
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

    def isSingular(self, pronouns: Text):
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

    def pronouns2gender(self, pronouns: Text):
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

    def process(self, message: Message, **kwargs: Any):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""

        pronouns = []
        tokens, poses = message.get("tokens"), message.get("poses")
        for i, (w, p) in enumerate(zip(tokens, poses)):
            if p == 'r' and self.legalPronouns(w):
                # 增加性别、单复数属性
                t_start = i
                t_end = i + 1
                start = get_start(i, tokens=tokens)
                end = start + len(w)
                pronouns.append({
                    'start': start,
                    'token_start': t_start,
                    'end': end,
                    'token_end': t_end,
                    'value': w,
                    'pos': p,
                    'gender': self.pronouns2gender(w),  # gender
                    'singular': self.isSingular(w),  # singular and the plural
                })
        message.set("pronouns", pronouns, add_to_output=True)

