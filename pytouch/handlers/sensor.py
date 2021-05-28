# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import cv2


class SensorHandler:
    def __init__(self, cv_device):
        self.sensor_cap = cv2.VideoCapture(cv_device)
        if not self.sensor_cap.isOpened():
            raise IOError("Could not open sensor capture device.")

    def get_frame(self):
        ret, frame = self.sensor_cap.read()
        if not ret:
            raise IOError("Could not read next frame.")
        return frame

    @property
    def dev(self):
        return self.sensor_cap
