---
id: touch
title: Touch Detection
sidebar_label: Touch Detection
---

The `TouchDetect` task predicts if there is an object in contact with the surface of the sensor.

### Available Pre-Trained Models

The following pre-trained models are available for use with the `TouchDetect` task:

| Dataset                           | Model Name           | Accuracy |
| --------------------------------- | -------------------- | -------- |
| DIGIT                             | `touchdetect_`       | xx.yy    |
| GelSight                          | `touchdetect_`       | xx.yy    |
| OmniTact                          | `touchdetect_`       | xx.yy    |
| Joint (DIGIT, Gelsight, OmniTact) | `joint_touchdetect_` | xx.yy    |

### Usage

Initialize the `TouchDetect` task with a sensor and pre-trained model,

```python
touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")
```

Calling `touch_detect` on an input sample predicts with certainty if there is an object touching the sensor surface.

```python
is_touching, certainty = touch_detect(sample)
```

### Normalization

The `TouchDetect` task loads default transform and normalization values from `pytouch.models.touch_detect.TouchDetectModelDefaults` and is suitable for any pre-trained model from the `TorchVision` package.

For custom models provide a custom class when initializing `TouchDetect` in the format of:

```python
touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues)

```

Where `MyTouchDetectValues` are in the following format,

```python
class MyTouchDetectValues:
    SCALES = [64, 64]
    MEANS = [0.123, 0.123, 0.123]
    STDS = [0.123, 0.123, 0.123]
    CLASSES = 2
```
