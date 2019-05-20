#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 15:33
# @Author  : zhangzhen
# @Site    : 
# @File    : relation.py
# @Software: PyCharm
import datetime
import json
import copy
import re
from typing import Dict, Any, Text
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message

from litemind.nlu.model import Metadata
from litemind.nlu.utils.coref import CorefHelper
from litemind.relation_extract.language import LanguageAnalysis


class RelationExtractor(Component):
    name = ""
    provides = []
    requires = []
    defaults = {}
    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(RelationExtractor, self).__init__(component_config)

        rule_path = component_config.get('rule_path')
        relation_path = component_config.get('relation_path')
        relation_reg_path = component_config.get('relation_reg_path')

        with open(rule_path, "r", encoding='utf-8') as f:
            content = f.read()
            self.rules = json.loads(content)['rules']

        with open(relation_path, "r", encoding='utf-8') as f:
            self.relation = {}.fromkeys([s.strip() for s in f.readlines()])

        with open(relation_reg_path, "r", encoding='utf-8') as f:
            self.relation_reg = [s.strip() for s in f.readlines()]

    def process(self, message: Message, **kwargs: Any):
        sentence = message.get("sentence")
        nodes = message.get("nodes")
        # 如果句子长度小于3，认为不包含关系信息
        relations = []
        if len(sentence) < 3:
            pass
        else:
            for item in self.rules:
                rel = self.compare(item["pattern"], item["relation"], nodes)
                for r in rel:
                    if r not in relations and self.valid_relation(r['rel']):
                        relations.append(r)
        message.set('relations', relations, add_to_output=True)

    def valid_relation(self, rel_name):
        if rel_name in self.relation:
            return 1
        for reg in self.relation_reg:
            if re.match(reg, rel_name, flags=re.IGNORECASE):
                return 1
        return 0

    def compare(self, pattern, relation, nodes):
        # pattern的示例：[{'s': {'word': '阿姨'}, 't': {'pos': '(n[a-z]{0,2}|r)'}, 'rel': 'ATT'},
        # {'s': {'pos': 'n[a-z]{0,2}'}, 't': 'con1.s', 'rel': 'ATT'}]

        if len(pattern) == 0:
            return []
        pattern = copy.deepcopy(pattern)
        path = []  # path为每次循环的候选node路径，如[[(5, 1)], [(11, 9)]]，每个元组中第1个元素表示s的位置，第二个元素表示t的位置
        depth = 0  # 用于记录匹配几个规则

        for p in pattern:
            p_bak = copy.deepcopy(p)
            depth += 1
            flag = False  # 是否找到p的标志s
            if len(path) > 0:  # 如果path已经有候选路径，需要考虑到路径的影响
                for pa in path:
                    if len(pa) < depth - 1:  # 不符合条件的匹配
                        continue
                    # 出现引用的情况，如con1.s，转换成loc位置
                    if not isinstance(p['s'], dict):
                        con = p['s'].split('.')
                        p_bak['s'] = {'loc': pa[int(con[0][-1]) - 1][0]} if con[1] == 's' else {
                            'loc': pa[int(con[0][-1]) - 1][1]}

                    if not isinstance(p['t'], dict):
                        con = p['t'].split('.')
                        p_bak['t'] = {'loc': pa[int(con[0][-1]) - 1][0]} if con[1] == 's' else {
                            'loc': pa[int(con[0][-1]) - 1][1]}

                    for node_t in nodes:
                        if node_t[3][0] == 0:  # 出发结点为root的情况
                            continue
                        node_s = nodes[node_t[3][0] - 1]

                        # 匹配是否满足条件，若满足，则加入loc_s, loc_t，否则继续搜索
                        if self.node_regular(node_t, node_s, p_bak):
                            pa.append((node_t[3][0], node_t[0]))
                            flag = True
            else:  # 如果path没有候选路径，则直接遍历
                for node_t in nodes:
                    if node_t[3][0] == 0:  # 出发结点为root的情况
                        continue
                    node_s = nodes[node_t[3][0] - 1]

                    # 匹配是否满足条件，若满足，则加入loc_s, loc_t，否则继续搜索
                    if self.node_regular(node_t, node_s, p):
                        path.append([(node_t[3][0], node_t[0])])
                        flag = True

            # 如果遍历完匹配不上p，表示没有符合模板的句式，返回[]；如果能匹配上，继续下一轮规则的匹配
            if not flag:
                return []

        out = []
        # 如果每轮p都能匹配上，意味着关系存在，输出最后结果
        for pa in path:
            if len(pa) < depth:  # 不符合条件的匹配
                continue
            if not isinstance(relation['s'], dict):
                con = relation['s'].split('.')
                s = nodes[pa[int(con[0][-1]) - 1][0] - 1][1] if con[1] == 's' else \
                    nodes[pa[int(con[0][-1]) - 1][1] - 1][1]
            else:
                s = relation['s']

            if not isinstance(relation['t'], dict):
                con = relation['t'].split('.')
                t = nodes[pa[int(con[0][-1]) - 1][0] - 1][1] if con[1] == 's' else \
                    nodes[pa[int(con[0][-1]) - 1][1] - 1][1]
            else:
                t = relation['t']

            if not isinstance(relation['rel'], dict):
                con = relation['rel'].split('.')
                rel = nodes[pa[int(con[0][-1]) - 1][0] - 1][1] if con[1] == 's' else \
                    nodes[pa[int(con[0][-1]) - 1][1] - 1][1]
            else:
                rel = relation['rel']
            out.append({'s': s, 't': t, 'rel': rel})
        return out

    @staticmethod
    def node_regular(node_t, node_s, p):
        # 匹配rel
        if node_t[3][1] != p['rel']:
            return 0
        # 匹配t结点
        if 'loc' in p['t']:
            if node_t[0] != p['t']['loc']:
                return 0
        if 'word' in p['t']:
            if node_t[1] not in p['t']['word']:
                return 0
        if 'pos' in p['t']:
            if not re.findall(p['t']['pos'], node_t[2]):
                return 0

        # 匹配s结点
        if 'loc' in p['s']:
            if node_s[0] != p['s']['loc']:
                return 0
        if 'word' in p['s']:
            if node_s[1] not in p['s']['word']:
                return 0
        if 'pos' in p['s']:
            if not re.findall(p['s']['pos'], node_s[2]):
                return 0
        return 1


if __name__ == '__main__':
    import pprint

    text = "小明生病了，他的阿姨王兰在照顾他"
    text = "百度公司的ceo是李彦宏"
    text = "李彦宏是百度公司的ceo"
    text = "张三是百度公司的会计"
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

    meta = model_metadata.for_component(index=2)
    relation = RelationExtractor.load(meta=meta, model_dir=model_dir, model_metadata=Metadata.load(model_dir))
    relation.process(message)

    pprint.pprint(message.data)
