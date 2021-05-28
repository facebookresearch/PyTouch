---
id: slip
title: Slip Detection
sidebar_label: Slip Detection
---

The `SlipDetect` task predicts if an object has slipped while being grasped.

### Available Pre-Trained Models

The following pre-trained models are available for use with the `SlipDetect` task:

| Dataset | Model Name            | Accuracy |
| ------- | --------------------- | -------- |
| DIGIT   | `slipdetect_resnet18` | xx.yy    |

### Input

Input into the `SlipDetect` task expects a `12` consecutive frames to predict if slip has occured.

### Usage

Initialize the `SlipDetect` task with a sensor and pre-trained model,

```python
slip_detect = SlipDetect(DigitSensor, zoo_model="slipdetect_resnet18")
```

### Normalization

The `SlipDetect` task loads default transform and normalization values from `pytouch.models.slip_detect.SlipDetectModelDefaults` and is suitable for any pre-trained model from the `TorchVision` package.

For custom models provide a custom class when initializing `SlipDetect` in the format of:

```python
slip_detect = SlipDetect(model=my_custom_model, defaults=MySlipDetectValues)

```

Where `MySlipDetectValues` are in the following format,

```python
class MySlipDetectValues:
    SCALES = [64, 64]
    MEANS = [0.123, 0.123, 0.123]
    STDS = [0.123, 0.123, 0.123]
```
