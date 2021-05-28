# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import ContactArea, TouchDetect


def multi_task():
    base_img_path = "./path/to/sample/img"
    sample_img_path = "./path/to/sample/img"

    base_img = ImageHandler(base_img_path).nparray
    sample_img = ImageHandler(sample_img_path).nparray
    sample_img_2 = sample_img.copy()

    # initialize with default configuration of ContactArea task
    pt = pytouch.PyTouch(DigitSensor, tasks=[ContactArea, TouchDetect])
    major, minor = pt.ContactArea(sample_img, base=base_img)
    is_touching = pt.TouchDetect(sample_img)

    print(f"Is touching? {is_touching}")
    print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))
    ImageHandler.save("surface_contact_1.png", sample_img)

    # initialize with custom configuration of ContactArea and TouchDetect task
    contact_area = ContactArea(base=base_img, contour_threshold=10)
    major, minor = contact_area(sample_img_2)

    touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")
    is_touching = touch_detect(sample_img)

    print(f"Is touching? {is_touching}")
    print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))
    ImageHandler.save("surface_contact_2.png", sample_img_2)


if __name__ == "__main__":
    multi_task()
