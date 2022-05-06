# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch


def interpolate_img(img, rows, cols):
    r"""
    Interpolates the image to new dimensions
        
    img: torch.Tensor (C x H x W)
    rows: Number of rows to interpolate to
    cols: Number of column to interpolate to
    """
    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)
    
    return img
