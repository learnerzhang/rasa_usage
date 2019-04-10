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

from litemind.nlu.utils import get_start

logger = logging.getLogger(__name__)


class LtpHelper(Component):
    """A new component"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "ltp"

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
        super(LtpHelper, self).__init__(component_config)
        self.path = component_config['path']
        self.lexicon = component_config['lexicon']

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
        message.set("tokens", list(self.segmentor.segment(message.text)), add_to_output=True)

    def extract_poses(self, message: Message):
        if message.get("tokens", default=None):
            self.extract_tokens(message)

        message.set("poses", list(self.postagger.postag(message.get("tokens"))))

    def extract_tagseq(self, message: Message):
        """
        实体抽取, 这部分需要扩张
        :param message:
        :return:
        """
        message.set("tagseq", list(self.recognizer.recognize(message.get("tokens"), message.get("poses"))))

    def extract_parses(self, message: Message):
        message.set("arcs", self.parser.parse(message.get("tokens"), message.get("poses")))

    def extract_labels(self, message: Message):
        message.set("labels", self.labeller.label(message.get("tokens"), message.get("poses"), message.get("arcs")))

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
        tokens, labels = message.get("tokens"), message.get("tagseq")
        i, start, end = 0, 0, 0
        entites = []
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
                    'dim': dim,
                    'start': _start,
                    'token_start': start,
                    'end': _end,
                    'token_end': i + 1,
                    'value': value
                }
                entites.append(ent)
                start = 0
            elif labels[i].startswith('B'):
                start = i
            elif labels[i].startswith('S'):
                dim = labels[i].split('-')[1]
                value = "".join(tokens[i:i + 1])

                _start = get_start(i, tokens=tokens)
                _end = _start + len(value)
                ent = {
                    'dim': dim,
                    'start': _start,
                    'token_start': i,
                    'end': _end,
                    'token_end': i + 1,
                    'value': value
                }
                entites.append(ent)
            else:  # O
                pass
            i += 1
        message.set("entities", entites, add_to_output=True)

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

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional = None,
             **kwargs):
        return cls(meta)
