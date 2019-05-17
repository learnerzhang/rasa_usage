from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import time
import copy
import logging
import warnings
import pprint
import regex as re
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message
from rasa.nlu.utils import read_json_file, write_json_to_file
from typing import Any, Text, Optional, Dict

from litemind.nlu.model import Metadata
from litemind.nlu.utils import duplicate

logger = logging.getLogger(__name__)


class CustomizedEntityExtractor(EntityExtractor):
    name = "ner_customized"

    output_provides = ["entities"]

    defaults = {
        "yml_path": None
    }

    def __init__(self, component_config=None, entity_customize=None):
        # type: (...) -> None
        super(CustomizedEntityExtractor, self).__init__(component_config)
        self.entity_customize = entity_customize if entity_customize else {}

        # independent of other entity
        self.independent_patterns = {}

        self.dependent_patterns = {}

        for dim, ent in self.entity_customize.items():
            patterns = ent['rules']
            _inde_patterns = [re.compile(p, re.I) for p in patterns if '@' not in p]
            if _inde_patterns:  # 不依赖的规则
                self.independent_patterns.update({dim: _inde_patterns})

            _de_patterns = []
            for p in patterns:
                # (?<=出生于)@dim:time,length:[4,5]
                if "@" in p:
                    item = {}

                    reg_tokens = p.split("@")
                    if len(reg_tokens) == 2:
                        reg_string = reg_tokens[0]  # reg_string
                        rare_string = reg_tokens[1]
                        item.update({"flag": True})
                    elif len(reg_tokens) == 3:
                        reg_string = reg_tokens[2]  # reg_string
                        rare_string = reg_tokens[1]
                        item.update({"flag": False})

                    kvs = rare_string.split(",")
                    for kv in kvs:
                        pair = kv.split(':')
                        if pair[0] == 'length':
                            length = pair[1][1:-1].split('-')
                            min = length[0]
                            max = length[1]
                            item.update({"min": int(min), "max": int(max)})
                        else:
                            item.update({pair[0]: pair[1]})

                    item.update({'pattern': re.compile(reg_string, re.I)})
                    # print(item)
                    _de_patterns.append(item)
            if _de_patterns:
                self.dependent_patterns.update({dim: _de_patterns})

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None
        import yaml
        yml_path = self.component_config["yml_path"]
        self.entity_customize = yaml.load(open(yml_path, encoding='utf-8'))

    def search_idx_entity(self, token, item, entities):
        t_end = token.end()
        t_start = token.start()
        flag = item['flag']
        t_dim = item['dim']
        idx = None
        for i, ent in enumerate(entities):
            start, end = int(ent['start']), int(ent['end'])
            if ent['entity'] == t_dim:
                if (flag and t_end == start) or (not flag and t_start == end):
                    idx = i
        return idx

    def process(self, message, **kwargs):
        extracted = []
        # 对简单规则实体规则扫描
        for dim, patterns in self.independent_patterns.items():
            for pattern in patterns:
                tokens = pattern.finditer(message.text)
                for token in list(tokens):
                    # print(token.start(), token.end(), token.group())
                    extracted.append(self.token2json(dim, token))

        # 对依赖实体规则扫描
        entites = message.get("entities", [])
        if entites:
            for dim, patterns in self.dependent_patterns.items():
                for pattern in patterns:
                    tokens = pattern['pattern'].finditer(message.text)
                    for token in list(tokens):
                        # TODO search entity
                        # print(token.start(), token.end(), token.group())
                        idx = self.search_idx_entity(token, pattern, entities=entites)
                        if idx is not None:
                            ent = entites[idx]
                            start, end = ent['start'], ent['end']
                            if pattern['min'] <= end - start + 1 <= pattern['max']:
                                ent['entity'] = dim
                                ent['extractor'] = self.name

        extracted = self.add_extractor_name(extracted)
        extracted = duplicate(extracted)
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    def token2json(self, dim, token):
        return {
            'text': token.group(),
            'entity': dim,
            'start': token.start(),
            'end': token.end(),
            'value': token.group()
        }

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        # type: (Text) -> Optional[Dict[Text, Any]]
        if self.entity_customize:
            entity_customize_file = os.path.join(model_dir, file_name)
            write_json_to_file(entity_customize_file, self.entity_customize, separators=(',', ': '))
            return {"customize_file": file_name}
        else:
            return {"customize_file": None}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional['DucklingHTTPExtractor'] = None,
             **kwargs: Any
             ):
        # type: (...) -> CustomizedEntityExtractor

        file_name = meta.get("customize_file")
        if not file_name:
            entity_customize = None
            return cls(meta, entity_customize)

        entity_customize_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entity_customize_file):
            entity_customize = read_json_file(entity_customize_file)
        else:
            entity_customize = None
            warnings.warn("Failed to load synonyms file from '{}'"
                          "".format(entity_customize_file))
        return cls(meta, entity_customize)


if __name__ == '__main__':
    logger.info("Start CustomizedEE")

    model_dir = 'models/entity/model_20190516-164503'
    model_metadata = Metadata.load(model_dir)
    meta = model_metadata.for_component(index=4)
    cee = CustomizedEntityExtractor.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))

    context = {}
    time = time.time()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}

    # """
    strings = [
        '张三出生日期2018年6月汉族男性 20岁',
        'A1309220900002014120003120223197105074079王林,维族，汉族出生于北京',
        '2019年1月30号放假',
        '张三出生于北京',
        '张三出生日期2019年12月',
        "其通过QQ81095335联系了一个卖手机的人林乐QQ：31685205.乐乐QQ2841672827.，被盗QQ号码：2881487340，QQ号码（2257982534），QQ号码(2257982234)了解了关于苹果6S手机的情况，谈好价钱，对方发来一条支付宝账号vi9523@163.com，收款人：张智安，并向其要地址、电话、联系方式通过快递发货，其将个人信息发给对方，向该账号打款5000元现金，对方发过来另一个QQ号2398937323称是发货部，其和该发货部联系，"
    ]

    for s in strings:
        message = Message(s, data=None, time=time)
        print(10 * "=")
        pprint.pprint(message.data)
        cee.process(message)
        pprint.pprint(message.data)

        print("\n")
    # """
    # s = "出生地为北京"
    # message = Message(s, data=default_output_attributes, time=time)
    # cee.process(message)
    # print(message)
