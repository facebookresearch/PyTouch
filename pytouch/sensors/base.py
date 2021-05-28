# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pytouch.common import SensorDataSources


class SensorBase:
    def __init__(self, sensor, data_source, data_path=None):
        self._sensor = sensor
        self.data_source = data_source
        self.data_path = data_path

        if data_source == SensorDataSources.DATASET and data_path is None:
            raise AssertionError(
                "Sensor data source specified with dataset but no dataset path provided."
            )
        if data_source == SensorDataSources.VIDEO and data_path is None:
            raise AssertionError(
                "Sensor data source specified with video but no dataset path provided."
            )

    def name(self):
        return self._sensor

    def __repr__(self):
        repr_str = "{0} with {1} data source".format(self._sensor, self.data_source)
        if self.data_path is not None:
            repr_str += ", using dataset {0}".format(self.data_path)
        return repr_str
