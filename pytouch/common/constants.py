# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from enum import Enum


class SensorDataSources(Enum):
    # Tensor or numerical input
    RAW = "raw"
    # Image Input
    IMAGE = "image"
    # DataLoader or file path with dataset
    DATASET = "dataset"
    # Video file
    VIDEO = "video"
    # Direct input from sensor
    SENSOR = "sensor"
    # Simulator input
    SIM = "simulator"
