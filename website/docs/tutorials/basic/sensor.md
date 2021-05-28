---
id: sensor_input
title: Sensor Input
sidebar_label: Sensor Input
---

PyTouch provides a sensor handler class which streams frames from sensors based on `cv2.VideoCapture`.

It allows for some basic functionality such as:

- Retrieving available data or frames from sensor

### Usage

An example is shown below on the usage of `SensorHandler`

```python
from pytouch.handlers import SensorHandler

cv_device = 0
my_sensor = SensorHandler(cv_device)
```

Sensor data is returned with,

```python
while True:
    frame = my_sensor.get_frame()
```
