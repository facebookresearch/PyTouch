# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class ImageHandler:
    def __init__(self, img_path, convert="RGB"):
        self.img = Image.open(img_path).convert(convert)
        self.convert = convert

    @staticmethod
    def tensor_to_PIL(self, img_tensor):
        img_tensor = img_tensor.squeeze_(0)
        return transforms.ToPILImage()(img_tensor).convert(self.convert)

    @property
    def tensor(self):
        return transforms.ToTensor()(self.img).unsqueeze_(0)

    @property
    def image(self):
        return self.img

    @property
    def nparray(self):
        return np.array(self.img)

    @staticmethod
    def save(file_name, img):
        if isinstance(img, Image.Image):
            # this is a PIL image
            img.save(file_name)
        else:
            # cv2 image
            cv2.imwrite(file_name, img)
