# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import pprint
import time
import random

from pytouch_sensors import digit
from pytouch_sensors.digit import Digit

logging.basicConfig(level=logging.DEBUG)

# Print a list of connected DIGIT's
digits = digit.find()
print("Connected DIGIT's to Host:")
pprint.pprint(digits)

# Connect to a Digit device with serial number with friendly name
digit = Digit("D01001", "Left Gripper")
digit.connect()

# Maximum value for each channel is 15
rgb_list = [(15, 0, 0), (0, 15, 0), (0, 0, 15)]

# digit.show_view()
import cv2

# Cycle through colors R, G, B
while True:
    rgb = (random.randint(0, 15), random.randint(0, 15), random.randint(0, 15))
    digit.set_intensity_rgb(*rgb)
    frame = digit.get_frame()
    cv2.imshow(f"Digit View {digit.serial}", frame)
    if cv2.waitKey(1) == 27:
        break
    time.sleep(1)

cv2.destroyAllWindows()

# Set all LEDs to same intensity
digit.set_intensity(15)

digit.disconnect()
