# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os

from torchvision import datasets
from torchvision.datasets import VisionDataset

_log = logging.getLogger(__name__)


class DigitFolder(VisionDataset):
    """DIGIT data loader where the images are arranged in this way: ::
        root/class0/serial_number/xxx.png
        root/class0/serial_number/xxy.png

        root/class1/serial_number/xxx.png
        root/class1/serial_number/xxy.png

        root/class2/serial_number/xxx.png
        root/class2/serial_number/xxy.png

    Args:
        exclude (list): Exclude DIGIT devices by specifying a list of serial numbers.
    """

    def __init__(
        self,
        root,
        transform=None,
        exclude=None,
        loader=datasets.folder.default_loader,
        baseline=None,
    ):
        super(DigitFolder, self).__init__(
            root, transform=transform, target_transform=None
        )
        self.exclude = exclude
        self.loader = loader

        classes, class_to_idx = self._get_classes(self.root)
        self.class_to_idx = class_to_idx

        samples = self._get_digits(self.root, self.exclude)
        self.classes = classes
        self.samples = samples
        self.serials = [serial[2] for serial in samples]

    def _get_classes(self, root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        class_to_idx = {class_name: x for x, class_name in enumerate(classes)}
        return classes, class_to_idx

    def _get_digits(self, root, exclude):
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
                    dirs[:] = list(filter(lambda x: x not in exclude, dirs))
                for fname in sorted(files):
                    if "baseline" not in fname:
                        path = os.path.join(class_dir, fname)
                        serial_number = path.split(os.path.sep)[-2]
                        item = path, class_idx, serial_number
                        digit_set.append(item)
        return digit_set

    def __getitem__(self, index):
        path, target, serial_number = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, serial_number

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        exclude_str = f"Exlcuded DIGITs: {', '.join(self.exclude)}"
        return "\n".join([super().__repr__(), exclude_str])
