# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

from torchvision import transforms

from ..utils.transforms import TemporalDownSample

_log = logging.getLogger(__name__)


class SlipDetect:
    def __init__(self, transform_data_cfg, checkpoint_path):
        pass

    @staticmethod
    def transform(data_cfg, train=False):
        transforms_list = []
        transforms_list.append(transforms.Scale(data_cfg.sample_size))
        transforms_list.append(transforms.CenterCrop(data_cfg.sample_size))
        if train:
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomVerticalFlip())
        transforms_list.append(transforms.ToTensor())
        if data_cfg.normalized_data:
            mean = data_cfg.mean
            std = data_cfg.std
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
        spatial_transform = transforms.Compose(transforms_list)
        temporal_transform = TemporalDownSample(data_cfg.sample_duration)
        target_transform = None
        return spatial_transform, temporal_transform, target_transform
