#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 4:03 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : cws_entity_extractor.py.py
# @Software: PyCharm
import datetime
import logging
import json
from typing import Dict, Text, Any, Optional

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message

from litemind.nlu.model import Metadata
from litemind.nlu.utils import HttpSessionContext

logger = logging.getLogger(__name__)


class CWSEntityExtractor(EntityExtractor):
    name = "ner_cws"

    provides = ['entities']

    defaults = {
        "url": None,
        "dimensions": None,
    }

    http_session = HttpSessionContext()

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(CWSEntityExtractor, self).__init__(component_config)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        self.request = self.http_session.open()
        # extracted = self.add_extractor_name(self.extract_entities(text, nlp))
        # logging.info('before cws entities:' + str(entities))
        extracted = self.extract_entities(message.text)
        # logging.info('after cws entities:' + str(extracted))
        self.request.close()

        extracted = self.add_extractor_name(extracted)
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, text):

        def column2json(body, dim, start, end, value):
            value = value if value and not value.startswith('$') else body
            return {
                "text": body,
                "entity": dim,
                "start": start,
                "end": end,
                "value": value
            }

        def get_start_index(result_text, pre_start=0):
            dimensions = self.component_config["dimensions"]
            arr = [(result_text.find(key, pre_start), value)
                   for key, value in dimensions.items() if result_text.find(key, pre_start) >= 0]
            return min(arr, key=lambda x: x[0]) if arr else (-1, None)

        # 中文分词接口请求格式与duckling不同,因此编码逻辑也不同,经过试验,这里需要预先转换成UTF-8格式
        pdata = str(text).encode('UTF-8')
        _entities = []
        try:
            result_request = json.loads(self.request.post(self._url(), pdata, timeout=3).text)
            logging.debug('cws zh raw return:' + str(result_request))
        except:
            logging.exception("request cws error, url:{} \ttext:{}".format(self._url(), text))
            return _entities
        result = result_request['recognize']

        start_index, dim = get_start_index(result)
        end = 0
        while start_index >= 0:
            # 左括号的位置
            end_index = result.find("}", start_index)
            body = result[start_index + 4: end_index]
            start = text.find(body, end)
            end = start + len(body)
            _entities.append(column2json(body, dim, start, end, body))
            start_index, dim = get_start_index(result, end_index)
        return _entities

    @classmethod
    def create(cls, component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'CWSEntityExtractor':
        return cls(component_config)

    def _url(self):
        return self.component_config.get("url")

    @classmethod
    def load(cls, meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['CWSEntityExtractpr'] = None,
             **kwargs: Any
             ):
        # type: (...) -> CWSEntityExtractpr
        return cls(meta)


if __name__ == '__main__':
    import pprint

    model_dir = './models/entity/model_20190516-164503'
    model_metadata = Metadata.load(model_dir)
    meta = model_metadata.for_component(index=2)
    cws = CWSEntityExtractor.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    text = "A1309261000002013060100120223197309195059邵忠升^2013年06月19日16时许，" \
           "赵海营（男，30岁，现住：肃宁县梁村镇赵官庄村，联系电话：15132719655）报警称：" \
           "今天上午7时许将大门其锁好外出，到下午16时许其回家发现自家大门敞开着，撒在院内的一条狼青狗被盗了。" \
           "被盗狼青狗特征：是一条1年半的狼青狗，高约60公分，长约80公分，现市场价值2000余元。" \
           "涉案总价值2000余元。案件性质关键词：撬门压锁。, 出生地北京，现居住在新疆"

    context = {}
    time = datetime.datetime.now()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}
    message = Message(text, data=default_output_attributes, time=time)

    pprint.pprint(message.data)
    cws.process(message)
    pprint.pprint(message.data)