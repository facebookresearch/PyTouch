# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

_log = logging.getLogger(__name__)


def _get_slip_videos_and_annotations(root, subset, prefix):
    with open(root, "r") as f:
        data_list = json.load(f)

    videos = []
    labels = []

    for obj in data_list:
        selected_subset = obj["subset"]
        if selected_subset == subset:
            if subset == "testing":
                # no label for testing data
                videos.append(os.path.join(prefix, obj["videopath"]))
            else:
                videos.append(os.path.join(prefix, obj["videopath"]))
                labels.append(obj["label"])
    return videos, labels


def _get_slip_labels(root, label_file_name="manual_frame.txt"):
    slip_label = []
    clip_list = None
    cnt = 0
    data_path = os.path.join(root, "data", label_file_name)

    for line in open(data_path, "r"):
        line = line.split(",")[0].split("    ")[-1]
        if cnt % 13 == 0:
            clip_list = []
        if (cnt % 13 > 0) and (cnt % 13 < 11):
            clip_list.append(int(line))
        if cnt % 13 == 11:
            slip_label.append(clip_list)
        cnt = cnt + 1
    slip_label = np.array(slip_label)
    return slip_label


class SlipVideoClip(Dataset):
    def __init__(
        self,
        root,
        prefix,
        spatial_transform=None,
        temporal_transform=None,
        default_loader=datasets.folder.default_loader,
        target_transform=None,
        subset="train",
        frame_duration=128,
    ):
        data_path = os.path.join(prefix, root)
        self.videos, self.annotations = _get_slip_videos_and_annotations(
            data_path, subset, prefix
        )
        self.slip_label = _get_slip_labels(prefix)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = default_loader
        self.default_frame_indicies = list(range(frame_duration))
        self.subset = subset
        self.frame_duration = frame_duration

    def __getitem__(self, index):
        sequence_path = os.path.join(self.root, self.videos[index])
        sequence_list = sequence_path.split("/")
        seq_num = int(sequence_list[-1].replace("seq", ""))
        obj_num = int(sequence_list[-2].replace("obj", ""))
        frame_indices = None

        if seq_num < 10:
            shift = np.random.choice(5, 1)[0]
            seq_begin = self.slip_label[obj_num][seq_num] + shift
            seq_end = seq_begin + self.frame_duration
            frame_indices = list(range(seq_begin, seq_end))
        else:
            shift = np.random.choice(100, 1)[0]
            frame_indices = list(range(shift, shift + self.frame_duration))

        # list of frames
        clip = self.loader(sequence_path, frame_indices)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.annotations[index]
        if self.target_transform is not None:
            target = self.target_transform
        return clip, target

    def __len__(self):
        return len(self.videos)


class SlipVideo(Dataset):
    def __init__(
        self,
        root,
        prefix,
        spatial_transform=None,
        temporal_transform=None,
        default_loader=datasets.folder.default_loader,
        target_transform=None,
        subset="train",
        frame_duration=128,
    ):
        data_path = os.path.join(prefix, root)
        self.videos, self.annotations = _get_slip_videos_and_annotations(
            data_path, subset, prefix
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = default_loader
        self.default_frame_indices = list(range(frame_duration))
        self.target_transform = target_transform
        self.subset = subset

    def __getitem__(self, index):
        sequence_path = os.path.join(self.root, self.videos[index])
        frame_indices = self.default_frame_indices
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        # list of frames
        clip = self.loader(sequence_path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.annotations[index]
        if self.target_transform is not None:
            target = self.target_transform
        return clip, target

    def __len__(self):
        return len(self.videos)
