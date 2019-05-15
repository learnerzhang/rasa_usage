#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-01 15:47
# @Author  : zhangzhen
# @Site    : 
# @File    : stg.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import Text, Optional, Any, Dict, List

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from gensim import models, utils
import numpy as np

from litemind.nlu.utils import pronouns2gender, span_output_format

logger = logging.getLogger(__name__)


class Strategy(Component):
    """
        匹配策略研究
    """
    name = "stg"
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(Strategy, self).__init__(component_config)
        self.w2v_path = component_config['w2v_path']
        self.model = models.KeyedVectors.load_word2vec_format(self.w2v_path)

    def similarity(self, word1, word2):
        """
        计算向量之间的欧氏距离
        :param word1:
        :param word2:
        :return:
        """
        return np.linalg.norm(self.word2vec(word1) - self.word2vec(word2))

    def word2vec(self, tokens):
        """
        文本->向量
        :param tokens:
        :return:
        """
        vec = []
        for token in tokens:
            try:
                vec.append(self.model[utils.to_unicode(token)])
            except KeyError:
                logging.warning("{} %s not found!".format(token))
        return np.sum(vec, axis=0) / len(vec)

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        pass

    def start(self, ent, tokens):
        for idx, token in enumerate(tokens):
            if ent['start'] == token['start']:
                return idx
        return -1

    def end(self, ent, tokens):
        for idx, token in enumerate(tokens):
            if ent['end'] == token['end']:
                return idx + 1
        return -1

    def multi_sample_class_entities(self, pronoun: Dict, entities: List[Dict], message: Message):
        """
        指代词在同类实体, 采用代词和实体支配词的相似度计算, 当大于0.5 阈值则关联,
        否则, 选取最近的实体
        :param pronoun:
        :param entities:
        :param message:
        :return:
        """
        tokens, segments, arcs = message.get("tokens"), message.get("segments"), message.get("arcs")

        if len(entities) == 0:
            return None
        elif len(entities) == 1:
            # print(p, "->", entities[top])
            return entities[0]
        else:
            # 其他策略选择
            pronoun_end = pronoun['end']
            # 考虑代词下面n个实体
            # TODO 注意边界范围
            if pronoun_end + 3 < len(message.text):
                ctx_pronoun = message.text[pronoun_end:pronoun_end + 3]
            elif pronoun_end + 2 < len(message.text):
                ctx_pronoun = message.text[pronoun_end:pronoun_end + 2]
            else:
                ctx_pronoun = message.text[pronoun_end:pronoun_end + 1]

            # TODO
            pronoun_arc = arcs[self.start(pronoun, tokens)]
            pronoun_dep_words = segments[pronoun_arc.head - 1]
            arc_sims = []
            ctx_sims = []
            for ent in entities:
                # context similarity
                # TODO  next context of ent
                ctx_ent = segments[self.end(ent, tokens)]
                ctx_sim = self.similarity(ctx_pronoun, ctx_ent)
                ctx_sims.append(ctx_sim)

                # TODO arcs similarity
                arc = arcs[self.start(ent, tokens)]
                ent_dep_words = segments[arc.head - 1]
                arc_sim = self.similarity(pronoun_dep_words, ent_dep_words)
                arc_sims.append(arc_sim)

            # 基于依赖词的相似度计算
            arc_indices = np.argsort(arc_sims)
            arc_top = arc_indices[0]
            arc_sec = arc_indices[1]

            # 基于实体环境相似
            ctx_indices = np.argsort(ctx_sims)
            ctx_top = ctx_indices[0]
            ctx_sec = ctx_indices[1]
            # 相似的依赖词
            if arc_sims[arc_top] <= 0.5 or arc_sims[arc_sec] - arc_sims[arc_top] >= 0.4:
                # print(p, "->", entities[top])
                return entities[arc_top]
            else:
                return entities[ctx_top]

    def stag(self, pronoun: Dict, message: Message):
        """
        1、 代词只前面的出现的实体
        2、 代词类别, 区分人物代词
        3、 多个人实体或者物事实体代词, 考虑依存词的情况下, 按就近原则
        4、 距离原则
        :param pronoun:
        :param message:
        :return:
        """
        p_text = message.text[pronoun['start']: pronoun['end']]
        spans = message.get("spans", [])

        # step1 解决前指现象
        entities = [span for span in spans if span['label'] != 'Pronoun']
        entities = [e for e in entities if e['end'] <= pronoun['start']]

        if len(entities) > 0:
            # step2 性别, 单复数现象
            if pronouns2gender(p_text) == '它':
                # 非人类指代: 它
                entities = [e for e in entities if e['label'] != 'PER']
            else:
                # 人实体指代: 他/她
                entities = [e for e in entities if
                            e['label'] == 'PER' and (e['gender'] == pronouns2gender(p_text) or e['gender'] == '未')]

            # step2 多个实体, 需要考虑就进原则, 支配词
            return self.multi_sample_class_entities(pronoun, entities, message)
        else:
            logging.warning(
                "{}[start={}, end={}] not match the mention entity".format(pronoun.get("value"), pronoun.get("start"),
                                                                           pronoun.get("end")))
            return None

        return None

    def process(self, message: Message, **kwargs: Any):

        spans = message.get("spans", [])
        pronouns = [span for span in spans if span['label'] == 'Pronoun']
        coreferences = []
        for pronoun in pronouns:
            ent = self.stag(pronoun, message)
            if ent:
                coreferences.append(
                    {
                        "pronoun": {'start': pronoun['start'], 'end': pronoun['end']},
                        "entity": {'start': ent['start'], 'end': ent['end']}
                    })
        span_output_format(spans)
        message.set("coreferences", coreferences, add_to_output=True)
        logging.info("coref data: {}".format(message.data))

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional = None,
             **kwargs):
        return cls(meta)
