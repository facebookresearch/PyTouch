# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytouch.datasets import SlipVideo, SlipVideoClip

_log = logging.getLogger(__name__)


class SlipDetectDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        video_clip=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.video_clip = video_clip

    def setup(self, stage=None):
        slip_dataset_fn = SlipVideoClip if self.video_clip else SlipVideo

        self.train_dataset = slip_dataset_fn(
            self.cfg.data.path,
            self.cfg.data.prefix,
            spatial_transform=self.cfg.data.spatial_transform,
            temporal_transform=self.cfg.data.spatial_transform,
            target_transform=self.cfg.data.target_transform,
            subset="train",
            frame_duration=self.cfg.data.frame_duration,
        )

        self.val_dataset = slip_dataset_fn(
            self.cfg.data.path,
            self.cfg.data.prefix,
            spatial_transform=self.cfg.data.spatial_transform,
            temporal_transform=self.cfg.data.spatial_transform,
            target_transform=self.cfg.data.target_transform,
            subset="validation",
            frame_duration=self.cfg.data.frame_duration,
        )

        self.dataset_len = len(self.train_dataset) + len(self.val_dataset)

        _log.info(
            f"Total dataset size: {self.dataset_len}, "
            + f"train {len(self.train_dataset)}, val {len(self.val_dataset)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.n_threads,
            pin_memory=self.cfg.training.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.n_threads,
            pin_memory=self.cfg.training.pin_memory,
            shuffle=False,
        )
