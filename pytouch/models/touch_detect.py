# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn
from torchvision import models


class TouchDetectModelDefaults:
    SCALES = [64, 64]
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    CLASSES = 2


class TouchDetectModel:
    def __init__(
        self,
        model=models.resnet18,
        state_dict=None,
        defaults=TouchDetectModelDefaults,
        **kwargs
    ):
        super(TouchDetectModel, self).__init__()
        self._model = model
        self.state_dict = state_dict
        self.defaults = defaults

        self._init_model(model, **kwargs)
        if state_dict is not None:
            self._load_state_dict()
        self._model.eval()

    def __call__(self, input):
        return self._model(input)

    def _init_model(self, model, **kwargs):
        if self._model.__name__ == "mobilenet_v2":
            self._model = model(**kwargs)
            self._model.classifier[1] = nn.Linear(
                self._model.last_channel, self.defaults.CLASSES
            )
        elif self._model.__name__ == "resnet18":
            self._model = model(**kwargs)
            self._model.fc = nn.Linear(
                self._model.fc.in_features, self.defaults.CLASSES
            )
        else:
            raise NotImplementedError()

    def _load_state_dict(self):
        self._model.load_state_dict(self.state_dict)
