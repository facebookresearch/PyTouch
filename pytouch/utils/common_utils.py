# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch


def flip(x):
    return torch.flip(x, dims=[0])


def min_clip(x, min_val):
    return torch.max(x, min_val)


def max_clip(x, max_val):
    return torch.min(x, max_val)


def normalize(x, min_val, max_val):
    return (x - torch.min(x)) * (max_val - min_val) / (
        torch.max(x) - torch.min(x)
    ) + min_val


def pandas_col_to_numpy(df_col):
    df_col = df_col.apply(
        lambda x: np.fromstring(
            x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),
            sep=", ",
        )
    )
    df_col = np.stack(df_col)

    return df_col


def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(
        arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),
        sep=", ",
    )
    return arr_npy
