# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch.optim as optim

_log = logging.getLogger(__name__)


def choose_optimizer(optim_name):
    if optim_name == "Adam":
        optimF = optim.Adam
    elif optim_name == "SGD":
        optimF = optim.SGD
    else:
        raise NotImplementedError
    _log.info(f"Using optimizer {optimF.__name__}")
    return optimF
