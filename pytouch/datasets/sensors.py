# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os

import torch
from torchvision import datasets
from torchvision.datasets import VisionDataset

_log = logging.getLogger(__name__)


class SensorFolder(VisionDataset):
    """Sensor data loader where the images are arranged in this way: ::
        root/class0/sensor_type/serial_number/xxx.png
        root/class0/sensor_type/serial_number/xxy.png

        root/class1/sensor_type/serial_number/xxx.png
        root/class1/sensor_type/serial_number/xxy.png

        root/class2/sensor_type/serial_number/xxx.png
        root/class2/sensor_type/serial_number/xxy.png

    Args:
        baseline: Filename pr efix for baseline image, example::
        baseline = "_baseline"
        Will reference all samples with filename "xxxyyyzzz_refimg.ext"

        exclude: Exclude sensor type and serials, example ::

        Exclude serial numbers matching,
        exclude = ["D00001", "D00002"]

        Exclude all DIGIT sensors,
        exclude = ["DIGIT"]
    """

    def __init__(
        self,
        root,
        transform=None,
        baseline="_baseline",
        exclude=None,
        loader=datasets.folder.default_loader,
    ):
        super(SensorFolder, self).__init__(
            root, transform=transform, target_transform=None
        )
        self.exclude = exclude if exclude else []
        self.loader = loader
        self.baseline = baseline

        classes, class_to_idx = self._get_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx

        samples = self._get_sensors(self.root, self.exclude)
        self.samples = samples
        self.sensor_types = [sample[2] for sample in samples]
        self.serials = [sample[3] for sample in samples]

    def _get_classes(self, root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        class_to_idx = {class_name: x for x, class_name in enumerate(classes)}
        return classes, class_to_idx

    def _get_baseline_path(self, path):
        ext_idx = path.rfind(".")
        baseline_sample_path = f"{path[:ext_idx]}{self.baseline}{path[ext_idx:]}"
        return baseline_sample_path

    def _exclude_sensors(self, dirs, exclude):
        if not isinstance(exclude, list):
            raise TypeError("exclude list must be a list")
        samples = list(filter(lambda x: x not in exclude, dirs))
        return samples

    def _get_sensors(self, root, exclude):
        digit_set = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[target_class]
            class_dir = os.path.join(root, target_class)
            for (
                class_dir,
                dirs,
                files,
            ) in os.walk(class_dir, topdown=True):
                if exclude is not None:
                    dirs[:] = self._exclude_sensors(dirs, exclude)
                for fname in sorted(files):
                    path = os.path.join(class_dir, fname)
                    # if self.baseline not in path:
                    sensor_type = path.split(os.path.sep)[-3]
                    serial_number = path.split(os.path.sep)[-2]
                    _log.debug(
                        f"Found sensor type {sensor_type} with serial {serial_number}"
                    )
                    item = path, class_idx, sensor_type, serial_number
                    digit_set.append(item)
        return digit_set

    def __getitem__(self, index):
        path, target, sensors_type, serial_number = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.baseline is not None:
            baseline_sample_path = self._get_baseline_path(path)
            baseline_sample = self.loader(baseline_sample_path)
            baseline_sample = self.transform(baseline_sample)
            sample = torch.cat((baseline_sample, sample), 2)
        print(f"{target} - {path} - {serial_number}")
        return sample, target, serial_number

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        exclude_str = f"Exlcuded DIGITs: {', '.join(self.exclude)}"
        return "\n".join([super().__repr__(), exclude_str])
