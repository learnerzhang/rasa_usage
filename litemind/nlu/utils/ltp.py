#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-29 18:07
# @Author  : zhangzhen
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from typing import Text, Optional, Any, Dict, List

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SentenceSplitter, SementicRoleLabeller

from litemind.nlu.utils import get_start, legalPronouns, pronouns2gender

logger = logging.getLogger(__name__)


class LtpHelper(Component):
    """A new component"""
    name = "ltp"

    provides = []

    requires = []

    defaults = {}

    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(LtpHelper, self).__init__(component_config)
        self.path = component_config['path']
        self.lexicon = component_config['lexicon']
        self.dimension = component_config['dimension']

        ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
        MODELDIR = os.path.join(ROOTDIR, self.path)
        self.segmentor = Segmentor()
        self.segmentor.load_with_lexicon(os.path.join(MODELDIR, "cws.model"), self.lexicon)

        self.postagger = Postagger()
        self.postagger.load(os.path.join(MODELDIR, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(MODELDIR, 'parser.model'))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(MODELDIR, "ner.model"))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(MODELDIR, "pisrl.model"))

    def extract_tokens(self, message: Message):
        segments = list(self.segmentor.segment(message.text))
        tokens = []
        start = 0
        for idx, segment in enumerate(segments):
            end = start + len(segment)
            tokens.append({'start': start, 'end': end})
            start = end

        message.set("segments", segments)
        message.set("tokens", tokens, add_to_output=True)

    def extract_poses(self, message: Message):
        if not message.get("segments", default=None):
            self.extract_tokens(message)

        message.set("poses", list(self.postagger.postag(message.get("segments"))))

    def extract_tagseq(self, message: Message):
        """
        实体抽取, 这部分需要扩张
        :param message:
        :return:
        """
        message.set("tagseq", list(self.recognizer.recognize(message.get("segments"), message.get("poses"))))

    def extract_parses(self, message: Message):
        message.set("arcs", self.parser.parse(message.get("segments"), message.get("poses")))

    def extract_labels(self, message: Message):
        message.set("labels", self.labeller.label(message.get("segments"), message.get("poses"), message.get("arcs")))

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        """Train this component.
       """
        pass

    def extract_entities(self, message: Message):

        # step1. 序列标注
        self.extract_tagseq(message)

        # step2.
        tokens, labels = message.get("segments"), message.get("tagseq")
        i, start, end = 0, 0, 0
        spans = []
        while i < len(labels):
            if labels[i].startswith('E'):
                dim = labels[i].split('-')[1]
                # 实体->词条
                value = "".join(tokens[start:i + 1])
                # 句子开始
                _start = get_start(start, tokens=tokens)
                # 句子结束
                _end = get_start(i, tokens=tokens) + len(value)
                ent = {
                    'label': self.dimension[dim],
                    'start': _start,
                    'end': _end,
                }
                spans.append(ent)
                start = 0
            elif labels[i].startswith('B'):
                start = i
            elif labels[i].startswith('S'):
                dim = labels[i].split('-')[1]
                value = "".join(tokens[i:i + 1])

                _start = get_start(i, tokens=tokens)
                _end = _start + len(value)
                ent = {
                    'label': self.dimension[dim],
                    'start': _start,
                    'end': _end,
                }
                spans.append(ent)
            else:  # O
                pass
            i += 1
        message.set("spans", spans, add_to_output=True)

    def extract_pronouns(self, message: Message, **kwargs: Any):
        pronouns = []
        tokens, poses = message.get("segments"), message.get("poses")
        for i, (w, p) in enumerate(zip(tokens, poses)):
            if p == 'r' and legalPronouns(w):
                # 增加性别、单复数属性
                start = get_start(i, tokens=tokens)
                end = start + len(w)
                pronouns.append({
                    'start': start,
                    'end': end,
                    'label': "Pronoun"
                })
        message.set("spans", message.get("spans", []) + pronouns, add_to_output=True)

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
        # 分词
        self.extract_tokens(message)
        # 词性标注
        self.extract_poses(message)
        # 句法依存
        self.extract_parses(message)

        # 抽取实体<序列标注+实体提取>
        self.extract_entities(message)

        # 抽取代词
        self.extract_pronouns(message)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional = None,
             **kwargs):
        return cls(meta)
