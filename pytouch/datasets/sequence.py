# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import VisionDataset

_log = logging.getLogger(__name__)


class ImageSequenceDataset(VisionDataset):
    """Image sequence data loader where the images are arranged in this way: ::
        root/0000/xxx.png
        root/0000/xxy.png

        root/0001/xxx.png
        root/0001/xxy.png

        Where the sequences are organized in their respective sequence sub-folders
        The dataset returns sorted sequences by sequence subfolder name, and by sorted
        file names in sequence subfolders.
    Args:
        root (str): path to sequence set folder
        img_shaders (list) [reserved]: Which shaders to include in the dataset
        real sensor images are only composed of color, whereas, TACTO images
        contain normal and depth images.
    """

    def __init__(
        self,
        root,
        img_shaders=["color"],
        transform=None,
        loader=datasets.folder.default_loader,
    ):
        super(ImageSequenceDataset, self).__init__(
            root, transform=transform, target_transform=None
        )
        self.root = root
        self.loader = loader
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.sequences, self.sequences_to_idx = self._get_sequences()
        self.imgs = self._get_imgs_in_sequence()

    def _get_sequences(self):
        sequences = [d.name for d in os.scandir(self.root) if d.is_dir()]
        sequences = sorted(sequences)
        sequences_to_idx = {seq_num: idx for idx, seq_num in enumerate(sequences)}
        return sequences, sequences_to_idx

    def _get_imgs_in_sequence(self):
        img_sequence = []
        for seq, idx in self.sequences_to_idx.items():
            img_seq_dir = os.path.join(self.root, seq)
            for f in os.listdir(img_seq_dir):
                path = os.path.join(img_seq_dir, f)
                item = path, idx
                img_sequence.append(item)
        return img_sequence

    def __getitem__(self, seq_idx):
        sequence_img_list = []
        sequence_paths = [img[0] for img in self.imgs if img[1] == seq_idx]
        for img in sequence_paths:
            sample = self.loader(img)
            if self.transform is not None:
                sample = self.transform(sample)
            sequence_img_list.append(sample)
        # img sequence list -> list of nd tesnsors S x C x H x W
        return torch.stack(sequence_img_list)

    def __len__(self):
        return len(self.sequences)

    def __repr__(self):
        sequence_info = (
            "Number of total sequence datapoints:"
            + f"{len(self.imgs)}, with {len(self.sequences)} unique sequences"
        )
        return "\n".join([super().__repr__(), sequence_info])
