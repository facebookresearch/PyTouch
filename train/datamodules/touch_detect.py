# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from pytouch.datasets import DigitFolder
from pytouch.tasks import TouchDetect

_log = logging.getLogger(__name__)


class TouchDetectDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.transform = TouchDetect.transform

    def setup(self, stage=None):
        train_dataset = DigitFolder(
            root=self.cfg.data.path,
            exclude=self.cfg.data.exclude,
            baseline=None,
            transform=self.transform(self.cfg.data.transform, train=True),
        )
        val_dataset = DigitFolder(
            root=self.cfg.data.path,
            exclude=self.cfg.data.exclude,
            baseline=None,
            transform=self.transform(self.cfg.data.transform, train=False),
        )

        self.dataset_len = len(train_dataset)
        dataset_idx = list(range(self.dataset_len))

        np.random.shuffle(dataset_idx)

        split_train_val = int(
            np.floor(self.cfg.training.train_val_ratio * self.dataset_len)
        )

        self.train_idx, self.val_idx = (
            dataset_idx[:split_train_val],
            dataset_idx[split_train_val:],
        )

        _log.info(
            f"Total dataset size: {self.dataset_len}, train {len(self.train_idx)}, val {len(self.val_idx)}"
            + f" using sensors {set(train_dataset.serials)}"
        )

        self.train_dataset = Subset(train_dataset, self.train_idx)
        self.val_dataset = Subset(val_dataset, self.val_idx)

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
