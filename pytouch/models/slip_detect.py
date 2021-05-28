# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn

from .slip_resnet import resnet18 as srn


class SlipDetectModelDefaults:
    SCALES = [1.0, 0.841, 0.707, 0.595, 0.500]
    MEANS = [0.374, 0.391, 0.377]
    STDS = [0.0750, 0.0751, 0.0736]
    CLASSES = 2


class SlipDetectModel(nn.Module):
    def __init__(
        self, model=srn, state_dict=None, defaults=SlipDetectModelDefaults, **kwargs
    ):
        super(SlipDetectModel, self).__init__()
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
        if self._model.__name__ == "resnet18":
            self._model = model(**kwargs)
            self._model.fc = nn.Linear(
                self._model.fc.in_features, self.defaults.CLASSES
            )
        else:
            raise NotImplementedError()

    def _load_state_dict(self):
        self._model.load_state_dict(self.state_dict)
