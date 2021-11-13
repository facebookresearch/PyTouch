# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
import torch.nn as nn
from torchvision import transforms

from pytouch.models import PyTouchZoo
from pytouch.models.touch_detect import TouchDetectModel, TouchDetectModelDefaults

_log = logging.getLogger(__name__)


class TouchDetect(nn.Module):
    def __init__(
        self,
        sensor,
        zoo_model="touchdetect_resnet",
        model_path=None,
        transform=None,
        defaults=TouchDetectModelDefaults,
        **kwargs
    ):
        self.sensor = sensor
        if "pretrained" not in kwargs:
            kwargs["pretrained"] = False
        self.model_path = model_path
        self.defaults = defaults
        self.transform = transform if transform is not None else self._transforms()

        if model_path is not None:
            # load custom model from path
            state_dict = PyTouchZoo.load_model(model_path)
        else:
            # load model from pytouch zoo
            zoo = PyTouchZoo()
            state_dict = zoo.load_model_from_zoo(zoo_model, sensor)
        self.model = TouchDetectModel(state_dict=state_dict)

    def __call__(self, frame):
        return self.is_touching(frame)

    def is_touching(self, frame):
        if isinstance(frame, torch.Tensor):
            output = self._predict(frame)
        else:
            output = self.process(frame)
            output = self._predict(output)
        return output

    def process(self, frame):
        frame_t = self.transform(frame)
        frame_t = frame_t.unsqueeze_(0)
        return frame_t

    def _predict(self, frame_t):
        with torch.no_grad():
            output = self.model(frame_t)
        prediction = output.data.cpu().numpy().argmax()
        certainty = torch.max(torch.max(nn.functional.softmax(output, dim=1)))
        return prediction, certainty

    def _transforms(self):
        transforms_list = [
            transforms.Resize(self.defaults.SCALES),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.defaults.MEANS, std=self.defaults.STDS),
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def transform(
        data_cfg, train=False, applyDefaultTransforms=False, customTransforms=None
    ):
        transforms_list = []
        transforms_list.append(transforms.Resize(list(data_cfg.scales)))
        if train:
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomVerticalFlip())
            # transforms_list.append(transforms.RandomRotation(degrees=5))
            # transforms_list.append(
            #     transforms.ColorJitter(
            #         brightness=0, contrast=0, saturation=0.1, hue=0.3
            #     )
            # )
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg.mean, std=data_cfg.std),
        ]
        frame_transform = transforms.Compose(transforms_list)
        _log.info("Transforms applied:")
        _log.info(frame_transform)
        return frame_transform
