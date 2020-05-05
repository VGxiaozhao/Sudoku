#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

import torch
import torch.nn.functional as F

from config import Config

_size = 28


class FontCNN(torch.nn.Module):
    def __init__(self):
        super(FontCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc3 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc4 = torch.nn.Linear(120, 84)
        self.fc5 = torch.nn.Linear(84, 10)

    def forward(self, x):
        y = x.view(-1, 1, _size, _size)
        y = F.max_pool2d(torch.relu(self.conv1(y)), 2)
        y = F.max_pool2d(torch.relu(self.conv2(y)), 2)
        y = y.view(-1, 16 * 5 * 5)
        y = torch.relu(self.fc3(y))
        y = torch.relu(self.fc4(y))
        y = self.fc5(y)
        return y


_model = FontCNN()
_model.to(Config.DEV)


_is_loaded = False


def predict(x):
    global _is_loaded
    if os.path.exists(Config.MODEL_DUMP) and not _is_loaded:
        _is_loaded = True
        _model.load_state_dict(torch.load(Config.MODEL_DUMP, map_location=Config.DEV))
    xb = torch.tensor(x, dtype=torch.float32, device=Config.DEV)
    _model.eval()
    with torch.no_grad():
        yb = _model(xb)
    ret = []
    for item in yb:
        ret.append(item.argmax())
    return ret, yb
