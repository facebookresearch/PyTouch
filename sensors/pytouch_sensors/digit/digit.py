# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import typing
import platform

try:
    import cv2
except ImportError as err:
    raise ImportError("Missing system dependencies:", err)

from dataclasses import dataclass

import numpy as np

try:
    import pyudev
except ImportError as err:
    pass
    # raise ImportError("Digit interface is only supported on Linux")

logger = logging.getLogger(__name__)


def _parse_digit(digit_dev: typing.Dict[str, str]) -> typing.Dict[str, str]:
    digit_info = {
        "dev_name": digit_dev.properties["DEVNAME"],
        "manufacturer": digit_dev.properties["ID_VENDOR"],
        "model": digit_dev.properties["ID_MODEL"],
        "revision": digit_dev.properties["ID_REVISION"],
        "serial": digit_dev.properties["ID_SERIAL_SHORT"],
    }
    return digit_info


def find(serial: typing.Optional[str] = None):
    context = pyudev.Context()
    digits = context.list_devices(subsystem="video4linux").match_attribute(
        "name", "DIGIT: DIGIT"
    )
    digits = [_parse_digit(_) for _ in digits]
    if serial:
        digits = next(filter(lambda k: k["serial"] == serial, digits), None)
    return digits


@dataclass(frozen=True)
class Stream:
    name: str
    resolution: typing.Tuple[int, int]
    fps: int


@dataclass(frozen=True)
class DigitStreams:
    QVGA_30fps = Stream("QVGA", (320, 240), 30)
    QVGA_60fps = Stream("QVGA", (320, 240), 60)
    VGA_15fps = Stream("VGA", (640, 480), 15)
    VGA_30fps = Stream("VGA", (640, 480), 30)


class Digit:
    __LIGHTING_SCALER = 17
    LIGHTING_MIN = 0
    LIGHTING_MAX = 15

    def __init__(
        self,
        serial: str,
        name: typing.Optional[str] = None,
    ) -> None:
        """
        DIGIT Device class for a single DIGIT
        :param serial: DIGIT device serial
        :param name: Human friendly identifier name for the device
        """
        self.serial: str = serial
        self.name: typing.Optional[str] = name
        self.__dev: cv2.VideoCapture = None

        self.dev_name: str
        self.manufacturer: str
        self.model: str
        self.revision: int

        self.resolution: typing.Tuple[int, int]
        self.fps: int
        self.intensity: typing.Tuple[int, int, int]

        self.populate(serial)

    def connect(
        self,
        stream=DigitStreams.QVGA_60fps,
        lighting: typing.Tuple[int, int, int] = None,
    ) -> None:
        logger.info(f"{self.serial}:Connecting to DIGIT")
        try:
            self.__dev = cv2.VideoCapture(self.dev_name)
        except cv2.error as err:
            logger.error("Cannot connect to DIGIT:", f"{self.serial} - {self.dev_name}")
            raise Exception(f"Error opening video stream {self.dev_name}:", err)

        logger.info(
            f"{self.serial}:Setting stream to {stream.name}",
            f"resolution: {stream.resolution}, fps: {stream.fps}",
        )
        self.set_resolution(stream.resolution)
        self.set_fps(stream.fps)
        if lighting:
            self.set_intensity_rgb(*lighting)
        else:
            self.set_intensity(self.LIGHTING_MAX)

    def _set_parameter(self, parameter: int, value: int) -> typing.Any:
        try:
            ret = self.__dev.set(parameter, value)
            return ret
        except Exception:
            raise IOError("Error accessing sensor, is the device connected?")

    def set_resolution(self, resolution: typing.Tuple[int, int]) -> None:
        """
        Sets stream resolution based on supported streams
        :param resolution: Stream resolution
        :return: None
        """
        self._set_parameter(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._set_parameter(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.resolution = resolution
        logger.debug(f"{self.serial}:Stream resolution set to {resolution}")

    def set_fps(self, fps: int) -> None:
        """
        Sets the stream fps, only valid values from Digit.STREAMS are accepted.
        This should typically be called after the resolution is set as the stream fps defaults to the
        highest fps
        :param fps: Stream FPS
        :return: None
        """
        self._set_parameter(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        logger.debug(f"{self.serial}:Stream FPS set to {self.fps}")

    def set_intensity(self, intensity: int) -> typing.Tuple[int, int, int]:
        """
        Sets all LEDs to specific intensity, this is a global control.
        :param intensity: Value between 0 and 15 where 0 is all LEDs off and 15 all
        LEDS full intensity
        :return: Returns the set intensity
        """
        if self.revision < 200:
            # Deprecated version 1.01 (1b) is not supported
            intensity = int(intensity / self.__LIGHTING_SCALER)
            logger.warn(
                "You are using a previous version of the firmware "
                "which does not support independent RGB control, update your DIGIT firmware."
            )
        self.intensity = self.set_intensity_rgb(intensity, intensity, intensity)
        return self.intensity

    def set_intensity_rgb(self, r: int, g: int, b: int) -> typing.Tuple[int, int, int]:
        """
        Sets LEDs to specific intensity, per LED control
        Perimitted values are between 0 (off/dim) and 15 (full brightness)
        :param r: Red value
        :param g: Green value
        :param b: Blue value
        :return: Returns the set intensity
        """
        if not all([x in range(0, self.LIGHTING_MAX + 1) for x in (r, g, b)]):
            raise ValueError(
                f"RGB values must be between {self.LIGHTING_MIN} and {self.LIGHTING_MAX}"
            )
        intensity = (r << 8) | (g << 4) | b
        self._set_parameter(cv2.CAP_PROP_ZOOM, intensity)
        self.intensity = (r, g, b)
        logger.debug(
            f"{self.serial}:LED intensity set to {intensity} (R: {r} G: {g} B: {b}"
        )
        return self.intensity

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
        Returns DIGIT device info
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
                f"\n\t- LED Intensity: {self.intensity}"
            )
        return info_string

    def populate(self, serial: str) -> None:
        """
        Find the connected DIGIT based on the serial number and populate device parameters into the class
        :param serial: DIGIT serial number
        :return:
        """
        digit = find(serial)
        if digit is None:
            raise Exception(f"Cannot find DIGIT with serial {self.serial}")
        self.dev_name = digit["dev_name"]
        self.manufacturer = digit["manufacturer"]
        self.model = digit["model"]
        self.revision = int(digit["revision"])
        self.serial = digit["serial"]

    def __repr__(self) -> str:
        return f"Digit(serial={self.serial}, name={self.name})"


__all__ = ["Digit", "DigitStreams", "find"]
