# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math

import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(image_set, class_names):
    images_iter = iter(image_set)
    x = y = math.ceil(math.sqrt(len(image_set)))
    fig = plt.figure(figsize=(x, y))
    for i in range(len(image_set)):
        image, label = images_iter.next()
        image, _ = image.numpy()
        image = image * 0.5 + 0.5
        image = np.transpose(image, (1, 2, 0))
        ax = fig.add_subplot(x, y, i + 1)
        ax.set_title(class_names[label])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(image)
    plt.tight_layout(True)
    plt.show()
