#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-29 11:13
# @Author  : zhangzhen
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import Text, Optional, Dict, Any

import os
import time
import torch
from rasa.nlu import utils
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from rasa.nlu.components import Component

from litemind.nlu.via import next_batch
from litemind.nlu.via.n2g import get_n2g

logger = logging.getLogger(__name__)


class N2G(nn.Module):
    def __init__(self, vocab2int, embed_size, class_num):
        """
        判断姓名 --> 性别的模型
        :param vocab2int:
        :param embed_size:
        :param class_num:
        """
        super(N2G, self).__init__()
        self.vocab2int = vocab2int
        self.embed = nn.Embedding(len(vocab2int) + 1, embed_size)
        self.fc = nn.Linear(embed_size, class_num)

    def forward(self, *input):
        x = self.embed(input[0])
        x = torch.mean(x, dim=1, keepdim=False)
        output = self.fc(x)
        output = F.log_softmax(output, dim=1)
        return output

    def predict(self, *input: str):
        rs = []
        if isinstance(input[0], list):
            for word in input[0]:
                x = torch.LongTensor([self.vocab2int[w] for w in word]).view(1, -1)
                output = self.forward(x)
                _, pred_y = torch.max(output, 1)
                rs.append(pred_y.numpy()[0])

        elif isinstance(input[0], str):
            x = torch.LongTensor([self.vocab2int[w] for w in input[0]]).view(1, -1)
            output = self.forward(x)
            _, pred_y = torch.max(output, 1)
            rs.append(pred_y.numpy()[0])
        return rs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class N2GHelper(Component):
    """A new component for n2g"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "n2g"
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

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 n2g: 'N2G' = None):
        super(N2GHelper, self).__init__(component_config)
        self.path = component_config['data_path']

        self.embed_size = component_config['embed_size']
        self.label2int = component_config['label2int']
        self.int2label = {v: k for k, v in self.label2int.items()}
        self.max_length = component_config['max_length']
        self.epoch = component_config['epoch']
        self.batch_size = component_config['batch_size']
        # print(self.int2label, self.label2int)
        # init via
        self.dat, self.tags, self.vocab2int = get_n2g(self.path, self.label2int, self.max_length)

        if n2g is not None:
            self.n2g = n2g
        else:
            self.n2g = N2G(self.vocab2int, self.embed_size, len(self.label2int))

    def train(self, training_data, cfg, **kwargs):
        """Train this component.

        This is the components chance to train itself provided
        with the training via. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""

        X_train, X_test, y_train, y_test = train_test_split(self.dat, self.tags, test_size=0.333, random_state=1234)
        optimizer = torch.optim.Adam(self.n2g.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        for e in range(self.epoch):
            for step, (b_x, b_y) in enumerate(next_batch(X_train, y_train, batch_size=self.batch_size)):
                inputs_x = torch.LongTensor(b_x)
                b_y = torch.LongTensor(b_y).squeeze()
                start = time.time()

                output = self.n2g(inputs_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    _, pred_y = torch.max(output, 1)
                    acc = sum(pred_y.numpy() == b_y.numpy()) / b_y.size(0)

                    print("{}/{} (epoch {}), train_loss = {:.3f}, acc = {:.3f}, time/batch = {:.3f}".format(step, int(
                        (len(y_train) + 1) / 256), e + 1, loss, acc, time.time() - start))
                if step % 1000 == 0:
                    test_inputs = torch.LongTensor(X_test)
                    test_labels = torch.LongTensor(y_test).squeeze()
                    test_output = self.n2g(test_inputs)
                    _, pred_y = torch.max(test_output, 1)
                    acc = sum(pred_y.numpy() == test_labels.numpy()) / test_labels.size(0)
                    print('Epoch:', e, '| train loss: %.4f' % loss.item(), 'test acc: %.4f' % acc)

    def process(self, message: Message, **kwargs):
        """Process an incoming message.
            判断人物实体的性别
            1. 提取人物实体
            2. 判读实体性别
        """
        entities = message.get("entities", [])
        for ent in entities:
            if ent['dim'] == 'Nh':
                """如果是人实体, 则..."""
                _g = self.n2g.predict(ent['value'])[0]
                ent.update({'gender': self.int2label[_g]})

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        # n2g model just for function test
        n2g_file = os.path.join(model_dir, 'n2g_params.pkl')
        self.n2g.save(n2g_file)

        # total object
        file_name = file_name + ".pkl"
        classifier_file = os.path.join(model_dir, file_name)
        utils.pycloud_pickle(classifier_file, self)
        return {"file": file_name}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['N2GHelper'] = None,
             **kwargs) -> 'N2GHelper':
        """Load this component from file."""

        file_name = meta.get("file")
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)
