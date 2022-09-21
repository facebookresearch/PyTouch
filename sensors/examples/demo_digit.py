# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import pprint
import time

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

# Print device info
print(digit.info())

# Change LED illumination intensity
digit.set_intensity(Digit.LIGHTING_MIN)
time.sleep(1)
digit.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"]
digit.set_resolution(qvga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit.set_fps(fps_30)

# Grab single frame from DIGIT
frame = digit.get_frame()
print(f"Frame WxH: {frame.shape[0]}{frame.shape[1]}")

# Display stream obtained from DIGIT
digit.show_view()

# Disconnect DIGIT stream
digit.disconnect()
