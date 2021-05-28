---
id: video_input
title: Video Input
sidebar_label: Video Input
---

PyTouch provides a video handler class which loads videos obtained from sensors.

It allows for some basic functionality such as:

- Resizing video
- Video information (frame position, and total frame count)
- Stepping through the video by consecutive frames
- Skipping frames or extract specific frames

### Usage

An example is shown below on the usage of `VideoHandler`

```python
from pytouch.handlers import VideoHandler

my_video = VideoHandler("/path/to/video.mp4")
```

Consecutive frames from the video can be obtained by,

```python
# Both frame and curent frame position are returned
frame, pos = my_video.get_frame()
```

The frame information can be queried and frame position set,

```python

current_frame, total_frames = my_video.frame

skip_frames = 10
next_frame = current_frame + skip_frames
if next_frame <= total_frames:
    my_video.set_frame_pos(next_frame)
    my_video.get_frame()
```

The video can be resized to return frames with a custom `(Width, Height)`,

```python
new_width = 64
new_height = 64
my_video = VideoHandler("/path/to/video.mp4", resize=(new_width, new_height))
```
