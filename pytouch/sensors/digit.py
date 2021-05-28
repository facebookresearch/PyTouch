# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from torchvision import transforms

from .base import SensorBase


class DigitSensorDefaults:
    SCALES = [64, 64]
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]


class DigitSensor(SensorBase):
    def __init__(
        self, data_source, data_path=None, transform=None, norm=DigitSensorDefaults
    ):
        super(DigitSensor, self).__init__(
            DigitSensor.__name__, data_source=data_source, data_path=data_path
        )
        self.transform = transform if transform else self.transform()

    def transform(self):
        transforms_list = [
            transforms.Resize(DigitSensorDefaults.SCALES),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DigitSensorDefaults.MEANS, std=DigitSensorDefaults.STDS
            ),
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def zoo_name():
        return "DigitSensor"
