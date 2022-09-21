# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import typing

import numpy as np

try:
    import cv2
except ImportError as err:
    raise ImportError("Missing system dependencies:", err)

try:
    import pyudev
except ImportError as err:
    print(
        "Gelsight Mini PyTouch interface is only supported on Linux, connect using GelsightMini(device_id=DEVICE) on Windows"
    )


logger = logging.getLogger(__name__)


def _parse_sensor(device: typing.Dict[str, str]) -> typing.Dict[str, str]:
    dev_str_lst = device.attributes.asstring("name").split(" ")
    devices_info = {
        "dev_name": device.properties["DEVNAME"],
        "manufacturer": dev_str_lst[0],
        "model": dev_str_lst[1],
        "revision": dev_str_lst[2],
        "serial": dev_str_lst[3][:-1],
    }
    return devices_info


def find(serial: typing.Optional[str] = None):
    context = pyudev.Context()
    devices = context.list_devices(subsystem="video4linux").match_attribute(
        "name", "GelSight Mini *"
    )
    devices = [_parse_sensor(_) for _ in devices]
    if serial:
        devices = next(filter(lambda k: k["serial"] == serial, devices), None)
    return devices


class GelsightMini:
    def __init__(
        self,
        serial: str,
        name: typing.Optional[str] = None,
        device_id: typing.Optional[str] = None,
    ) -> None:
        """
        Gelsight Mini device class
        :param serial: Gelsight Mini device serial
        :param name: Human friendly identifier name for the device
        """
        self.serial: str = serial
        self.name: typing.Optional[str] = name
        self.__dev: cv2.VideoCapture = None

        self.dev_name: str
        self.manufacturer: str
        self.model: str
        self.revision: str

        self.resolution: typing.Tuple[int, int]
        self.fps: int
        self.intensity: typing.Tuple[int, int, int]

        if device_id is not None:
            self.dev_name = device_id
            self.manufacturer = "GelSight"
            self.model = "Mini"
            self.revision = "N/A"
        else:
            self.populate(serial)

    def connect(
        self,
    ) -> None:
        logger.info(f"{self.serial}:Connecting to Gelsight Mini")
        try:
            self.__dev = cv2.VideoCapture(self.dev_name)
        except cv2.error as err:
            logger.error(
                "Cannot connect to Gelsight Mini:", f"{self.serial} - {self.dev_name}"
            )
            raise Exception(f"Error opening video stream {self.dev_name}:", err)

        res_width = self.__dev.get(cv2.CAP_PROP_FRAME_WIDTH)
        res_height = self.__dev.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.resolution = (res_width, res_height)
        self.fps = self.__dev.get(cv2.CAP_PROP_FPS)

    def get_frame(self, transpose: bool = False) -> np.ndarray:
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        ret, frame = self.__dev.read()
        if not ret:
            logger.error(
                f"Cannot retrieve frame data from {self.serial}, is device open?"
            )
            raise Exception(
                f"Unable to grab frame from {self.serial} - {self.dev_name}!"
            )
        if not transpose:
            frame = cv2.transpose(frame, frame)
            frame = cv2.flip(frame, 0)
        return frame

    def save_frame(self, path: str) -> np.ndarray:
        """
        Saves a single image frame to host
        :param path: Path and file name where the frame shall be saved to
        :return: None
        """
        frame = self.get_frame()
        logger.debug(f"Saving frame to {path}")
        cv2.imwrite(path, frame)
        return frame

    def get_diff(self, ref_frame: np.ndarray) -> np.ndarray:
        """
        Returns the difference between two frames
        :param ref_frame: Original frame
        :return: Frame difference
        """
        diff = self.get_frame() - ref_frame
        return diff

    def show_view(self, ref_frame: np.ndarray = None) -> None:
        """
        Creates OpenCV named window with live view of sensor, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """
        while True:
            frame = self.get_frame()
            if ref_frame is not None:
                frame = self.get_diff(ref_frame)
            cv2.imshow(f"Sensor View {self.serial}", frame)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def disconnect(self) -> None:
        logger.debug(f"{self.serial}:Closing device")
        self.__dev.release()

    def info(self) -> str:
        """
        Returns device info
        :return: String representation of device
        """
        has_dev = self.__dev is not None
        is_connected = False
        if has_dev:
            is_connected = self.__dev.isOpened()
        info_string = (
            f"Name: {self.name} Device: {self.dev_name}"
            f"\n\t- Model: {self.model}"
            f"\n\t- Revision: {self.revision}"
            f"\n\t- Connected: {is_connected}"
        )
        if is_connected:
            info_string += (
                f"\nStream Info:"
                f"\n\t- Resolution: {self.resolution}"
                f"\n\t- FPS: {self.fps}"
            )
        return info_string

    def populate(self, serial: str) -> None:
        """
        Find the connected Gelsight Mini based on the serial number and populate device parameters into the class
        :param serial: Gelsight Mini serial number
        :return:
        """
        sensor = find(serial)
        if sensor is None:
            raise Exception(f"Cannot find Gelsight Mini with serial {self.serial}")
        self.dev_name = sensor["dev_name"]
        self.manufacturer = sensor["manufacturer"]
        self.model = sensor["model"]
        self.revision = sensor["revision"]
        self.serial = sensor["serial"]

    def __repr__(self) -> str:
        return f"GelsightMini(serial={self.serial}, name={self.name})"


__all__ = ["GelsightMini", "find"]
