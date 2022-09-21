# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import pprint
import time

from pytouch_sensors import gelsight
from pytouch_sensors.gelsight import GelsightMini

logging.basicConfig(level=logging.DEBUG)

print("Connected Gelsight sensors to host:")
gelsight_sensors = gelsight.find()
pprint.pprint(gelsight_sensors)

gs_mini = GelsightMini("28CF-95MN", "Thumb")
gs_mini.connect()

# Print device info
print(gs_mini.info())

frame = gs_mini.get_frame()
print(f"Frame WxH: {frame.shape[0]}{frame.shape[1]}")

gs_mini.show_view()

gs_mini.disconnect()
