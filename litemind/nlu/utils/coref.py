#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 15:50
# @Author  : zhangzhen
# @Site    : 
# @File    : coref.py
# @Software: PyCharm
import datetime
from typing import Dict, Text, Any, Optional

import jieba.posseg as jposseg
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
import os
import json
import logging
from litemind.nlu.model import Metadata
from litemind.nlu.utils import HttpSessionContext

logger = logging.getLogger(__name__)


class CorefHelper(Component):
    name = ""
    provides = []
    requires = []
    defaults = {

    }
    language_list = None

    http_session = HttpSessionContext()

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(CorefHelper, self).__init__(component_config)

    def _url(self):
        return self.component_config.get("url")

    def train(self, training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        pass

    def process(self, message: Message, **kwargs: Any):

        self.request = self.http_session.open()
        # extracted = self.add_extractor_name(self.extract_entities(text, nlp))
        chains = self.extract_coref(message.text)
        self.request.close()

        if not chains:
            message.set("sentence", message.text)
        else:
            target_words = []
            flatten_chains = []
            for chain in chains:
                # 计算出实体词
                words = [s['mention'] for s in chain]  # [pair('该', 'r'), pair('员工', 'n')]
                auxiliary_words = words[0]  # 备选的实体词
                for i in range(1, len(words)):
                    flags = [s.flag for s in list(jposseg.cut(words[i]))]
                    if flags[0] != 'r' and len(words[i]) > len(auxiliary_words):
                        auxiliary_words = words[i]
                target_words.append(auxiliary_words)

                # 拼接字典并排序
                for s in chain:
                    s.update({'replace': auxiliary_words})
                flatten_chains.extend(chain)

            flatten_chains.sort(key=lambda x: x['start'], reverse=True)
            temp_sentence = list(message.text)

            for d in flatten_chains:
                temp_sentence[d['start']: d['end']] = d['replace']

            sentence = ''.join(temp_sentence)
            message.set("sentence", sentence)

    def extract_coref(self, text):
        pdata = {'text': str(text).encode('UTF-8')}
        chains = []
        try:
            result_request = json.loads(self.request.post(self._url(), json=pdata, timeout=3).text)
            logging.debug('coref zh raw return:' + str(result_request))
        except:
            logging.exception("request http coref error, url:{} \ttext:{}".format(self._url(), text))
            return chains

        if "corefChains" in result_request:
            chains = result_request['corefChains']
        return chains

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


if __name__ == '__main__':
    import pprint

    model_dir = './models/relation/model_20190517-161309'
    model_metadata = Metadata.load(model_dir)
    meta = model_metadata.for_component(index=0)
    coref = CorefHelper.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    text = "小明生病了，他的阿姨王兰在照顾他"

    context = {}
    time = datetime.datetime.now()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}
    message = Message(text, data=default_output_attributes, time=time)
    pprint.pprint(message.data)
    coref.process(message)
    pprint.pprint(message.data)
