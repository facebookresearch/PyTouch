# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from torchvision import transforms

from .base import SensorBase


class GelsightSensorDefaults:
    SCALES = [64, 64]
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]


class GelsightSensor(SensorBase):
    def __init__(self, data_source, data_path=None, transform=None):
        super(GelsightSensor, self).__init__(
            GelsightSensor.__name__, data_source=data_source, data_path=data_path
        )
        self.transform = transform if transform else self.transform()

    def transform(self):
        transforms_list = [
            transforms.Resize(GelsightSensorDefaults.SCALES),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=GelsightSensorDefaults.MEANS, std=GelsightSensorDefaults.STDS
            ),
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def zoo_name():
        return "GelSightSensor"
