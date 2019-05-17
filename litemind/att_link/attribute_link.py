#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-16 14:10
# @Author  : zhangzhen
# @Site    : 
# @File    : attribute_link.py.py
# @Software: PyCharm
import datetime
import logging
import pprint
import re
from collections import defaultdict
from typing import Dict, Text, Any, Optional
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa.nlu.training_data import Message

from litemind.nlu.model import Metadata
from litemind.nlu.utils.ltp import LtpHelper

logger = logging.getLogger(__name__)


class AttributeLink(Component):
    name = ""
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(AttributeLink, self).__init__(component_config)
        self.group_entity = component_config['group_entity']

    def process(self, message: 'Message', **kwargs: Any):
        semantic = message.get('semantic')
        tokens, entity_selected, valid_pos = message.get('tokens'), message.get('entity_selected'), message.get('valid_pos')
        logging.debug("tokens: {}".format(tokens))
        logging.debug("semantic: {}".format(semantic))

        assert len(tokens) == len(semantic)  # 保证分词结果与语义分析结果长度相等
        # 将COO和ATT同种关系的属性进行归类分组
        valid_relation = ['COO', 'ATT', 'SEO']
        valid_pos_reg = '(n[a-z]{0,2}|ws|r|i|m|j|a|b)'

        # valid_index的下标从1开始，与句法分析的下标一致
        valid_index = [index for index, word, pos, _ in semantic if re.match(valid_pos_reg, pos)]
        entity_index = [index for index, word, pos, _ in semantic if pos == 'nh']
        logging.debug("valid index:{}".format(valid_index))

        groups = defaultdict(list)  # 记录已命名的分组，如{'小明': [1, 2, 4], '小王': [6, 8]}
        unnamed_groups = []
        for i in valid_index:
            cur = semantic[i - 1]  # 当前的词
            p = cur[3][0]  # 表示依赖词的位置，如2
            r = cur[3][1]  # 表示'HED'等句法关系
            # 先对有效的词性的节点的两端进行初步分组，有以下几种情况

            if r in valid_relation:
                if cur[0] in entity_index:
                    # 第一种情况：属于有效关系，当前词是nh，依赖词是nh，这时要分成两组
                    if p in entity_index:
                        groups[cur[1]].extend([i])
                    # 第二种情况：属于有效关系，当前词是nh，依赖词不是nh，这时要分成同一组
                    else:
                        groups[cur[1]].extend([i, p])
                else:
                    # 第三种情况：属于有效关系，当前词不是nh，依赖词是nh，这时要分成同一组
                    if p in entity_index:
                        groups[semantic[p - 1][1]].extend([i, p])
                    # 第四种情况：属于有效关系，当前词不是nh，依赖词不是nh，这时要分成同一组，放入未命名
                    else:
                        unnamed_groups.append([i, p])
            else:
                if cur[0] in entity_index:
                    # 第五种情况：属于无效关系，当前词是nh
                    groups[cur[1]].extend([i])
                else:
                    # 第六种情况：属于无效关系，当前词不是nh
                    unnamed_groups.append([i])

        logging.debug('First groups: {}'.format(groups))
        logging.debug('First unnamed groups: {}'.format(unnamed_groups))

        # 以下将没有标签的属性归类到有标签的属性中，然后合并有交叉元素的若干组
        combined_unnamed_groups = self.combine_group(unnamed_groups)
        logging.debug('combined_groups: {}'.format(combined_unnamed_groups))
        for index, group in groups.items():
            temp_group = [group]
            temp_group.extend(combined_unnamed_groups)
            result = self.combine_group(temp_group, [0])
            groups.update({index: result[0]})
            combined_unnamed_groups = result[1:]

        combined_groups = list(groups.values())
        combined_groups.extend(combined_unnamed_groups)
        logging.debug('combined_groups: {}'.format(combined_groups))
        # 实体过滤
        logging.debug('valid_pos: {}'.format(valid_pos))

        filtered_groups = []
        for g in combined_groups:
            new_g = [i for i in g if i in valid_pos]
            if new_g:
                filtered_groups.append(new_g)
        logging.debug('filtered_groups: {}'.format(filtered_groups))
        logging.debug('entity_selected: {}'.format(entity_selected))

        # 判断是人，还是车
        grouped_entities = {'person': [], 'vehicle': []}
        for g in filtered_groups:
            # stat用来统计属性是人还是车？如果属于人，则人+1，如果属于车，则车+1，两者都有，各加0.5
            stat = {}.fromkeys(self.group_entity.keys(), 0)
            for i in g:
                cur_type = entity_selected[i]['entity']
                if cur_type in self.group_entity['person']:
                    if cur_type in self.group_entity['vehicle']:
                        stat['person'] += 0.5
                        stat['vehicle'] += 0.5
                    else:
                        stat['person'] += 1
                else:
                    if cur_type in self.group_entity['vehicle']:
                        stat['vehicle'] += 1

            score = max(stat.values())
            if score >= 1:
                if score == stat['vehicle']:
                    new_g = []
                    for i in g:
                        if entity_selected[i]['entity'] in self.group_entity['vehicle']:
                            new_g.append(entity_selected[i])
                    grouped_entities['vehicle'].append(new_g)
                if score == stat['person']:
                    new_g = []
                    for i in g:
                        if entity_selected[i]['entity'] in self.group_entity['person']:
                            new_g.append(entity_selected[i])
                    grouped_entities['person'].append(new_g)

        # 到此为止，完成person和vehicle的分类及属性过滤，{'person': [[1, 3, 6], [21, 23]], 'vehicle': [[28, 29]]}

        # 以下判断哪些属性是no_used
        # 如果最终有返回，则从entity_selected中删除，剩下的元素就是no_used
        used_index = []
        for group in grouped_entities.values():
            for g in group:
                for e in g:
                    used_index.append(e['index'])
        logging.debug('used_index: '.format(used_index))
        no_used = {k: v for k, v in entity_selected.items() if k not in used_index}

        logging.debug('no_used: {}'.format(no_used))
        grouped_entities.update({'ext': [list(no_used.values())]})
        message.set("grouped_entities", grouped_entities, add_to_output=True)



    @classmethod
    def create(cls, component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'AttributeLink':
        return cls(component_config)

    @classmethod
    def load(cls, meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['AttributeLink'] = None,
             **kwargs: Any
             ):
        # type: (...) -> AttributeLink
        return cls(meta)

    @staticmethod
    def combine_group(a_list, cur_index=None):
        # type: (List)->List
        # 如果有重复元素，会对列表中的分组进行合并，如[[1,2], [2,4], [5]]合并成[[1,2,4], [5]]
        b = [set(s) for s in a_list]
        if not cur_index:
            cur_index = list(range(len(b)))  # 保存更新的位置

        while True:
            last_index = cur_index[:]
            cur_index = []
            for i in range(len(b) - 1):
                if i in last_index:
                    for j in range(i + 1, len(b)):
                        if b[i].intersection(b[j]):
                            b[i] = b[i].union(b[j])
                            cur_index.append(i)
                            b[j] = set()
            if not cur_index:
                break
        b = [list(s) for s in b if len(s) > 0]
        return b


if __name__ == '__main__':
    entities = [{'entity': 'person', 'body': '小明', 'start': 0, 'end': 2},
                {'entity': 'gender', 'body': '男', 'start': 3, 'end': 4},
                {'entity': 'height', 'body': '180cm', 'start': 7, 'end': 12},
                {'entity': 'time', 'body': '上个月', 'start': 13, 'end': 16},
                {'entity': 'station', 'body': '北京站', 'start': 17, 'end': 20},
                {'entity': 'trip', 'body': 'G22', 'start': 21, 'end': 24},
                {'entity': 'city', 'body': '新疆', 'start': 25, 'end': 27},
                {'entity': 'age', 'body': '30岁', 'start': 34, 'end': 37},
                {'entity': 'person', 'body': '小黑', 'start': 38, 'end': 40},
                {'entity': 'color', 'body': '白色', 'start': 47, 'end': 49},
                {'entity': 'car_brand', 'body': '法拉利', 'start': 49, 'end': 52}
                ]

    text = '小明，男，身高180cm，上个月去北京站坐G22到新疆，与他同行的有30岁的小黑，他们开着一辆白色法拉利逃跑。'

    time = datetime.datetime.now()
    default_output_attributes = {"intent": {"name": None, "confidence": 0.0}, "entities": []}
    message = Message(text, data=default_output_attributes, time=time)
    message.set('entities', entities)

    # TODO
    model_dir = './models/link/model_20190517-131354'
    model_metadata = Metadata.load(model_dir)

    jieba_meta = model_metadata.for_component(index=0)
    jie = JiebaTokenizer.load(meta=jieba_meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))
    pprint.pprint(message.data)
    jie.process(message)

    ltp_meta = model_metadata.for_component(index=5)
    ltp = LtpHelper.load(meta=ltp_meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))
    ltp.process(message)

    meta = model_metadata.for_component(index=6)
    attlink = AttributeLink.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))
    pprint.pprint(message.get("entities"))

    attlink.process(message=message)
    pprint.pprint(message.get('grouped_entities'))
