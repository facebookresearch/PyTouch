# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class ContactArea:
    def __init__(
        self, base=None, draw_poly=True, contour_threshold=100, *args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        (
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
        ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            self._draw_major_minor(
                target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
            )
        return (major_axis, major_axis_end), (minor_axis, minor_axis_end)

    def _diff(self, target, base):
        diff = (target * 1.0 - base) / 255.0 + 0.5
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        return diff_abs

    def _smooth(self, target):
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        return diff_blur

    def _contours(self, target):
        mask = ((np.abs(target) > 0.04) * 255).astype(np.uint8)
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
        self,
        target,
        poly,
        major_axis,
        major_axis_end,
        minor_axis,
        minor_axis_end,
        lineThickness=2,
    ):
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        for contour in contours:
            if len(contour) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end
