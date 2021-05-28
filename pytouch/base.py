# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pytouch.common import SensorDataSources
from pytouch.sensors.base import SensorBase


class PyTouch:
    def __init__(
        self, sensor, source=SensorDataSources.RAW, source_path=None, tasks=None
    ):
        sensor_class = sensor if isinstance(sensor, type) else type(sensor)
        if not issubclass(sensor_class, SensorBase):
            raise AssertionError("An invalid sensor was specified.")

        self.sensor = sensor
        self.source = source
        self.source_path = source_path
        self.tasks = tasks

        # self.hanlder = self.init_handler()
        self.init_sensor()

        if tasks:
            self.init_tasks()

    def init_sensor(self):
        if isinstance(self.sensor, type):
            self.sensor = self.sensor(data_source=SensorDataSources.RAW)

    def init_handler(self):
        pass

    def init_tasks(self):
        for task in self.tasks:
            if isinstance(task, type):
                print(self.sensor)
                task = task(sensor=self.sensor)
            setattr(self, task.__class__.__name__, task)
