# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random


class TemporalDownSample(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        begin_index = random.randint(0, 4)
        out = frame_indices[begin_index::4]
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out
