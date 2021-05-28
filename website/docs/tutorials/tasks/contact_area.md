---
id: contact_area
title: Contact Area Estimation
sidebar_label: Contact Area Estimation
---

The `ContactArea` task estimates the major and minor axis of the surface contact.

A base image sample, `base` is provided when initializing the task. This base image sample is an image without any contact applied to the surface of the sensor. This base image is used for all future estimations unless changed.

### Usage

Calling the `ContactArea` task with a sample from the sensor returns the major and minor axis of the estimated contact and draws directly to the sample the contact ellipse.

This can be initialized through `pytouch.tasks.ContactArea` such as:

```python
from pytouch.tasks import ContactArea

base_img = ...
sample_img = ...
contact_area = ContactArea(base=base_img)

major, minor = contact_area(sample_img)
```

An example of saving `sample_img` after calling `contact_area(sample_img)`,

![Contact Area Example](/img/contact/example.png)
