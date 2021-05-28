# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect


def touch_detect():
    source = ImageHandler("/path/to/image")

    # initialize with task defaults
    pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
    is_touching, certainty = pt.TouchDetect(source.image)

    # initialize with custom configuration of TouchDetect task
    touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")

    is_touching, certainty = touch_detect(source.image)
    print(f"Is touching? {is_touching}, {certainty}")


if __name__ == "__main__":
    touch_detect()
