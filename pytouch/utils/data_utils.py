# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch


def interpolate_img(img, rows, cols):
    """
    img: C x H x W
    """
    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)
    return img
