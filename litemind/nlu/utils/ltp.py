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

import datetime
import logging
import os
from typing import Text, Optional, Any, Dict, List

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa.nlu.training_data import Message, TrainingData

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SentenceSplitter, SementicRoleLabeller

from litemind.nlu.utils import get_start, legalPronouns, pronouns2gender, tokenizer_extract

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
        tokens = list(self.segmentor.segment(message.text))
        segments = []
        start = 0
        for idx, token in enumerate(tokens):
            end = start + len(token)
            segments.append({'start': start, 'end': end})
            start = end
        message.set("segments", segments)
        message.set("tokens", tokens)

    def extract_poses(self, message: Message):
        if not message.get("tokens", default=None):
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
        tokens, poses = message.get("tokens"), message.get("poses")
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

    def entity_segment(self, message: 'Message', **kwargs: Any):
        # type: (List, List[Dict])->List
        """ 属性链接

        :param tokens: [word, word, word]
        :param entities: [{'entity': 'A', 'body': 'word', 'start': 0, 'end': 1}, ...]
        :return: [word, word, word]
        """
        entities = message.get("entities")
        tokens = message.get("tokens")

        if len(entities) == 0:
            return tokens
        else:
            # 求出tokens中所有词的starts和ends的坐标
            lengths = [len(w) for w in tokens]
            pos = [0]
            for p in lengths:
                pos.append(p + pos[-1])
            starts = pos[:-1]
            ends = pos[1:]

            #  标注长度和位置信息
            i = 0
            for e in entities:
                e['length'], e['index'] = e['end'] - e['start'], i
                i += 1

            # 保证entities的start和end，在starts和ends里面，否则筛除
            valid_entities = [e for e in entities if (e['start'] in starts) and (e['end'] in ends)]
            token_entities = [{'entity': w, 'body': w, 'start': start, 'end': end}
                              for w, start, end in zip(tokens, starts, ends)]
            # 对entities按长度的降序排列，意味着如果位置相同，长词语优先保留
            valid_entities.sort(key=lambda x: x['length'], reverse=True)
            valid_entities.extend(token_entities)
            valid_entities.sort(key=lambda x: x['start'], reverse=False)
            # 筛选实体，如有包含，较长的实体优先；如有交叉，先出现的实体优先；如完全相同，取第1个（意味着随机）
            p = 0
            filtered_entities = []

            for e in valid_entities:
                if e['start'] == p:
                    filtered_entities.append(e)
                    p = e['end']
            # 改变token
            word_tokens = [message.text[e['start']:e['end']]for e in filtered_entities]

            # 记录词语的位置
            entity_selected = {}
            i = 1
            for e in filtered_entities:
                if 'length' in e:
                    e.update({'index': i})
                    entity_selected.update({i: e})
                i += 1

            valid_pos = list(entity_selected.keys())

            message.set("tokens", word_tokens)
            message.set("entity_selected", entity_selected)
            message.set("valid_pos", valid_pos)

    def link_analyze(self, message: Message, **kwargs: Any):

        tokens = message.get("tokens", [])
        postags = message.get("poses", [])
        arcs = message.get("arcs")
        arcs = [(arc.head, arc.relation) for arc in arcs]
        semantic = list(zip(list(range(1, len(tokens) + 1)), tokens, postags, arcs))
        logging.debug('semantic structrue: {}'.format(semantic))
        # 以下是特殊情况下的句法调整
        # 第一种情况：记录动词“是”和“为”的位置
        loc = []
        for struc in semantic:
            if (struc[1] in ['是', '为']) and (struc[2] == 'v'):
                loc.append(struc[0])
        for i in loc:
            pre_loc = 0
            suf_loc = 0
            for j in range(1, i):
                if (semantic[j - 1][3][0] == i) and (semantic[j - 1][3][1] == 'SBV'):
                    pre_loc = j
            for j in range(i + 1, min(len(semantic) + 1, i + 10)):  # 最多间隔10个词语，对于宾语来说已经足够
                if (semantic[j - 1][3][0] == i) and (semantic[j - 1][3][1] == 'VOB'):
                    suf_loc = j
            if pre_loc and suf_loc:
                semantic[pre_loc - 1] = (
                    semantic[pre_loc - 1][0], semantic[pre_loc - 1][1], semantic[pre_loc - 1][2], (suf_loc, 'SEO'))

        # 第二种情况：此处是句法分析出错的情况，将实体识别成谓语成分SBV，词性为i
        loc = []
        for struc in semantic:
            if struc[2] == 'i':
                loc.append(struc[0])
        for i in loc:
            for j in range(1, i):
                if (semantic[j - 1][3][0] == i) and (semantic[j - 1][3][1] == 'SBV'):
                    semantic[j - 1] = (semantic[j - 1][0], semantic[j - 1][1], semantic[j - 1][2], (i, 'SEO'))

        # 第三种情况：记录动词“名叫”和“叫”的位置
        loc = []
        for struc in semantic:
            if (struc[1] in ['名叫', '叫', '叫做']) and (struc[2] == 'v'):
                loc.append(struc[0])
        for i in loc:
            for j in range(i + 1, min(len(semantic) + 1, i + 10)):
                if (semantic[j - 1][3][0] == i) and (semantic[j - 1][3][1] == 'VOB'):
                    semantic[j - 1] = (
                        semantic[j - 1][0], semantic[j - 1][1], semantic[j - 1][2], (semantic[i - 1][3][0], 'SEO'))

        message.set('semantic', semantic, add_to_output=False)

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
        # TODO 分词, 如果利用其它分词组件, 需要进一步调整
        if not message.get("tokens", default=None):
            self.extract_tokens(message)
            # 词性标注
            self.extract_poses(message)
            # 句法依存
            self.extract_parses(message)
            # 抽取实体<序列标注+实体提取>
            self.extract_entities(message)
            # 抽取代词
            self.extract_pronouns(message)
        else:
            # rasa tokenizers
            tokens = message.get("tokens")
            message.set("tokenizers", tokens)
            # List tokens
            tokens = [tokenizer_extract(token) for token in tokens]
            message.set("tokens", tokens)
            self.extract_poses(message)
            # 句法依存
            self.extract_parses(message)
            # 抽取实体<序列标注+实体提取>
            # 语义分割 ->
            self.entity_segment(message)
            # 属性分析 ->
            self.link_analyze(message)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional = None,
             **kwargs):
        return cls(meta)


if __name__ == '__main__':
    import pprint

    text = "A1309261000002013060100120223197309195059邵忠升^2013年06月19日16时许，" \
           "赵海营（男，30岁，现住：肃宁县梁村镇赵官庄村，联系电话：15132719655）报警称：" \
           "今天上午7时许将大门其锁好外出，到下午16时许其回家发现自家大门敞开着，撒在院内的一条狼青狗被盗了。" \
           "被盗狼青狗特征：是一条1年半的狼青狗，高约60公分，长约80公分，现市场价值2000余元。" \
           "涉案总价值2000余元。案件性质关键词：撬门压锁。, 出生地北京，现居住在新疆"
    text = "小明，男，身高180cm，上个月去北京站坐G22到新疆，与他同行的有30岁的小黑，他们开着一辆白色法拉利逃跑。"
    context = {}
    time = datetime.datetime.now()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}
    message = Message(text, data=default_output_attributes, time=time)

    model_dir = './models/link/model_20190517-113416'
    model_metadata = Metadata.load(model_dir)

    jieba_meta = model_metadata.for_component(index=0)
    jie = JiebaTokenizer.load(meta=jieba_meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    pprint.pprint(message.data)
    jie.process(message)

    ltp_meta = model_metadata.for_component(index=5)
    ltp = LtpHelper.load(meta=ltp_meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    pprint.pprint(message.data)
    ltp.process(message)
    pprint.pprint(message.data)
