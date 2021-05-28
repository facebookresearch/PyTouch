---
id: image_input
title: Image Input
sidebar_label: Image Input
---

PyTouch provides an image handler class which loads images obtained from sensors.

It allows for some basic functionality such as:

- Converting from `PIL.Image` to `Tensor`
- Converting from `Tensor` to `PIL.Image`
- Converting to `numpy.nparray`
- Saving `PIL` or `OpenCV` images

### Usage

An example is shown below on the usage of `ImageHandler`

```python
from pytouch.handlers import ImageHandler

my_image = ImageHandler("/path/to/image.png")
```

Returning the image,

```python
img = my_image.image
```

Returing a `Tensor`,

```python
img_tensor = my_image.tensor
```

Returning a `numpy array`,

```python
img_nparray = my_image.nparray
```

Converting from `Tensor` to `PIL.Image`,

```python
new_image = ImageHandler.tensor_to_PIL(img_tensor)
```

Saving to file,

```
ImageHandler.save("my_file_name.png", img_nparray)
# or
ImageHandler.save("my_file_name.png", img)
```
