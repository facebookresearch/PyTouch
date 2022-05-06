# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch


def flip(x):
    r"""
    Utility function. It takes tensor as a parameter.
    Returns reversed order of a n-D tensor along given axis.
    Dims is predefined as a list/tuple of an axis to flip on, dims=[0].
    """
    return torch.flip(x, dims=[0])


def min_clip(x, min_val):
    r"""
    Utility function. It takes two parameters.
    x is tensor, min_val is an axis, and you look for the maximum value along a particular axis. 
    The utility function returns the maximum value of a n-D tensor along a particular axis, 
    together with the indices corresponding to the maximum values. 
    """
    return torch.max(x, min_val)


def max_clip(x, max_val):
    r"""
    Utility function. It takes two parameters.
    x is tensor, max_val is an axis, and you look for the minimum value along a particular axis. 
    The utility function returns the minimum value of a n-D tensor along a particular axis, 
    together with the indices corresponding to the minimum values. 
    """
    return torch.min(x, max_val)


def normalize(x, min_val, max_val):
    r"""
    Utility function. It takes tensor and two integers with values between min_val and max_val.
    Uses formula for Min-Max Normalization with arbitrary set of values [a, b].
    We might choose to scale the normalized data in the range [0,1] or [0, 255]. 
    - torch.max() function returns the maximum value of a tensor
    - torch.min() function returns the minimum value of a tensor
    Normalizing image inputs ensures similar data distribution.
    """
    return (x - torch.min(x)) * (max_val - min_val) / (
        torch.max(x) - torch.min(x)
    ) + min_val


def pandas_col_to_numpy(df_col):
    r"""
    Converts selected column in Pandas dataframe to numpy array.
    """
    df_col = df_col.apply(
        lambda x: np.fromstring(
            x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),
            sep=", ",
        )
    )
    df_col = np.stack(df_col)

    return df_col


def pandas_string_to_numpy(arr_str):
    r"""
    Converts string to numpy array in pandas dataframe. 
    Replace function will not work on list.
    Data separation is done by comma. 
    """
    arr_npy = np.fromstring(
        arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),
        sep=", ",
    )
    return arr_npy
