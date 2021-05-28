---
id: intro
title: Tutorial Intro
sidebar_label: Introduction
---

import useBaseUrl from '@docusaurus/useBaseUrl'; import Link from '@docusaurus/Link';

## Intializing PyTouch

PyTouch can be intialized via two methods, simple and advanced. The simple initializing of PyTouch loads all pre-defined models and defaults for the selected tasks. Whereas, the advanced mode enables granular control of parameters.

### Simple

```python
import pytouch as pt

experiment = pt.PyTouch(pt.sensors.DigitSensor, task=[pt.tasks.TouchDetect])

...

is_touching = experiment.TouchDetect(my_input)
```

### Advanced

```python
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect

# Apply global transforms to all tasks
digit_sensor = DigitSensor(transform=my_transforms_compose)

touch_detect = TouchDetect(digit_sensor, model_path="/path/to/custom/model/")

...

is_touching = touch_detect(my_input)
```
