#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 16:35
# @Author  : zhangzhen
# @Site    : 
# @File    : language.py
# @Software: PyCharm
import datetime
import json
from typing import Dict, Text, Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message

from litemind.nlu.model import Metadata
from litemind.nlu.utils import HttpSessionContext, get_start
from pyltp import Segmentor, Postagger, Parser, SentenceSplitter, NamedEntityRecognizer
import os
import re
import logging

from litemind.nlu.utils.coref import CorefHelper

logger = logging.getLogger(__name__)


class LanguageAnalysis(Component):
    name = ""
    provides = []
    requires = []
    defaults = {

    }
    language_list = None

    http_session = HttpSessionContext()

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(LanguageAnalysis, self).__init__(component_config)
        self.dimensions = component_config['dimensions']
        ltp_path = component_config.get('ltp_path')

        self.postagger = Postagger()
        self.postagger.load(os.path.join(ltp_path, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(ltp_path, "parser.model"))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(ltp_path, "ner.model"))

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        pass

    def extract_words(self, text: Text):
        segment_url = self.component_config.get('segment_url')
        try:
            result_request = json.loads(self.request.post(segment_url, data=text.encode('utf-8'), timeout=3).text)
            logging.debug('cws zh segment raw return:' + str(result_request))
        except:
            logging.exception("request cws error, url:{} \ttext:{}".format(segment_url, text))

        words = result_request['segment'].split()
        return words

    def extract_recognizes(self, text: Text):
        recognize_url = self.component_config.get('recognize_url')
        try:
            result_request = json.loads(self.request.post(recognize_url, data=text.encode('utf-8'), timeout=3).text)
            logging.debug('cws zh recognize_url raw return:' + str(result_request))
        except:
            logging.exception("request cws error, url:{} \ttext:{}".format(recognize_url, text))

        ner = result_request['recognize']
        return ner

    @staticmethod
    def reseg(words, ner):
        # text: 平安集团CEO马明哲出席了会议
        # words:['在','北京', '平安', '集团', 'CEO', '马', '明哲', '出席', '了', '会议']
        # ner: 在{NT 北京平安集团}CEO{NR 马明哲}出席了会议
        # return: ['平安集团', 'CEO', '马明哲', '出席', '了', '会议']
        out_words = []
        loc2 = 0
        flag = False  # 标志是否在实体内部
        current_s = ''
        for i in words:
            next_loc = ner.find(i, loc2)
            if next_loc == loc2:
                if not flag:
                    loc2 += len(i)
                    out_words.append(i)
                else:
                    loc2 += len(i)
                    current_s += i
            else:
                if re.match('\{[A-Z]{1,3}\s', ner[loc2:]):
                    flag = True
                    current_s = i
                    loc2 = next_loc + len(i)
                elif ner[loc2] == '}':
                    if re.match('\}\s*{[A-Z]{1,3}\s', ner[loc2:]):
                        out_words.append(current_s)
                        loc2 = next_loc + len(i)
                        current_s = i
                    else:
                        flag = False
                        out_words.append(current_s)
                        loc2 = next_loc + len(i)
                        current_s = ''
                        out_words.append(i)
                else:
                    if not flag:
                        loc2 = next_loc + len(i)
                        out_words.append(i)
                    else:
                        current_s += i
                        loc2 = next_loc + len(i)

        return out_words

    def extract_entities(self, message: Message, **kwargs: Any):
        def column2json(body, dim, start, end, value):
            value = value if value and not value.startswith('$') else body
            return {
                "label": dim,
                "start": start,
                "end": end,
            }

        def get_start_index(result_text, pre_start=0):
            arr = [(result_text.find(key, pre_start), value)
                   for key, value in self.dimensions.items() if result_text.find(key, pre_start) >= 0]
            return min(arr, key=lambda x: x[0]) if arr else (-1, None)

        text_ner = message.get('text_ner')
        spans = []

        start_index, dim = get_start_index(text_ner)
        end = 0
        while start_index >= 0:
            # 左括号的位置
            end_index = text_ner.find("}", start_index)
            body = text_ner[start_index + 4: end_index]
            start = message.text.find(body, end)
            end = start + len(body)
            ent = column2json(body, dim, start, end, body)

            spans.append(ent)
            start_index, dim = get_start_index(text_ner, end_index)

        message.set("spans", spans, add_to_output=True)

    def process(self, message: Message, **kwargs: Any):

        # TODO 取消分句处理
        sentence = message.get('sentence')
        self.request = self.http_session.open()
        words = self.extract_words(text=sentence)
        ner = self.extract_recognizes(text=sentence)
        text_ner = self.extract_recognizes(text=message.text)
        logging.debug("response ner text: {}".format(text_ner))
        words = self.reseg(words, ner)
        postags = self.postagger.postag(words)  # 词性标注
        arcs = self.parser.parse(words, postags)  # 句法分析
        arcs = [(arc.head, arc.relation) for arc in arcs]
        nodes = list(zip(list(range(1, len(words) + 1)), words, postags, arcs))
        message.set("nodes", nodes)
        message.set("text_ner", text_ner)
        self.request.close()
        # origin Text
        self.extract_entities(message)


if __name__ == '__main__':
    import pprint

    text = "小明生病了，他的阿姨王兰在照顾他"
    context = {}
    time = datetime.datetime.now()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}
    message = Message(text, data=default_output_attributes, time=time)
    pprint.pprint(message.data)

    model_dir = './models/relation/model_20190520-134127'
    model_metadata = Metadata.load(model_dir)
    meta = model_metadata.for_component(index=0)
    coref = CorefHelper.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))
    coref.process(message)

    meta = model_metadata.for_component(index=1)
    coref = LanguageAnalysis.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    coref.process(message)
    pprint.pprint(message.data)
