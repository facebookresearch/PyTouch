# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random
from collections import OrderedDict

import numpy as np
import torch

_log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_dataset_info(dataset):
    print(f"Dataset size: {len(dataset)}")
    print(f"With {len(dataset.classes)} classes.")


def freeze_weights(nn_model):
    for name, param in nn_model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def convert_state_dict_if_from_pl(checkpoint):
    if "state_dict" not in checkpoint:
        _log.debug("Checkpoint is not a PyTorch-Lightning saved model.")
        return checkpoint
    else:
        _log.debug("Checkpoint is a PyTorch-Lightning saved model, extracting.")
        pl_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k
            if name.startswith("model."):
                name = name.replace("model.", "")
                pl_state_dict[name] = v
        return pl_state_dict


def _choose_last_fc_mode(mode):
    assert mode in ["score", "feature"]
    if mode == "score":
        last_fc = True
    elif mode == "feature":
        last_fc = False
    else:
        raise NotImplementedError

    _log.info(f"Setting fc mode to {mode}")
    return last_fc
