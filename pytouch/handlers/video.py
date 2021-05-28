# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import cv2


class VideoHandler:
    def __init__(self, path, resize=None):
        self.path = path
        self.is_resized = False
        self.current_frame = 0
        self.total_frames = 0

        self.video_cap = self.load_video(self.path)

        if resize is not None:
            self.resize(resize)

    def load_video(self, path):
        video_cap = cv2.VideoCapture(path)
        if not video_cap.isOpened():
            raise IOError("Could not open specified video source.")

        self.total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        ret, frame = video_cap.read()
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width, height, _ = frame.shape
        self.width = width
        self.height = height
        return video_cap

    def resize(self, resize=None, reset=False):
        if not self.video_cap.isOpened():
            raise AssertionError("VideoHandler must be initialized before resizing")
        if isinstance(resize, (tuple, list)) and (len(resize) == 2):
            width, height = tuple(map(int, resize))
            self.is_resized = True
        elif isinstance(resize, float):
            width, height, _ = self.video_cap.shape
            width = width / resize
            height = height / resize
            self.is_resized = True
        elif reset:
            width, height = self.width, self.height
            self.is_resized = False
        else:
            raise NotImplementedError(
                "Video resizing must specify a tuple, list or float for scaling factor."
            )

        self.resized_width = width
        self.resized_height = height

    def get_frame(self):
        ret, frame = self.video_cap.read()
        if not ret:
            raise IOError("Could not read next frame.")
        if self.is_resized:
            frame = cv2.resize(frame, (self.resized_width, self.resized_height))
        self.current_frame = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
        return (frame, self.current_frame)

    def set_frame_pos(self, pos):
        if pos > self.total_frames:
            raise AssertionError(
                "Specified frame position exceeds total frames available."
            )
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    @property
    def frame(self):
        return (self.current_frame, self.total_frames)
