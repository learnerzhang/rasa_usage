#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-10 14:59
# @Author  : zhangzhen
# @Site    : 
# @File    : lite.py
# @Software: PyCharm
from typing import Any, Dict, Text
import collections
from rasa.nlu.emulators import NoEmulator


class LinkEmulator(NoEmulator):
    def __init__(self) -> None:
        self.name = "link"

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return data

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""
        if 'intent' in data:
            del data['intent']

        if 'project' in data:
            del data['project']

        if 'model' in data:
            del data['model']

        if 'adapter' in data:
            del data['adapter']

        if 'entities' in data:
            entities = data.get('entities')
            del data['entities']
        else:
            entities = []

        if 'grouped_entities' in data:
            grouped_entities = data['grouped_entities']
            del data['grouped_entities']
        else:
            grouped_entities = {}

        spans = [{'start': ent['start'], 'end': ent['end'], 'label': ent['entity']} for ent in entities if
                 ent['entity'] in grouped_entities.keys()]
        span2idx = {span['label']: i for i, span in enumerate(spans)}

        attributes = []
        for entity, atts in grouped_entities.items():
            if entity in span2idx:
                eid = span2idx.get(entity)
            else:
                eid = -1
            attributes.extend(
                [{'eid': eid, 'attr_value': a['text'], 'attr_label': a['entity'], 'start': a['start'], 'end': a['end']}
                 for att in atts for a in att])
        data.update({'spans': spans})
        data.update({'attributes': attributes})

        return data
