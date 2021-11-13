# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import logging
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

_log = logging.getLogger(__name__)


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


"""
Optical flow visualization functions
"""


def flow_to_color(flow_uv, cvt=cv2.COLOR_HSV2BGR):
    hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_color = cv2.cvtColor(hsv, cvt)
    return flow_color


def flow_to_arrows(img, flow, step=8):
    img = copy.deepcopy(img)
    # img = (255 * img).astype(np.uint8)

    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = 5.0 * flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, color=(0, 255, 0), thickness=1)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img


def depth_to_color(depth):
    gray = (
        np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1) * 255
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def visualize_flow_cv2(
    img1, img2, flow_arrow=None, flow_color=None, win_size=(360, 360)
):
    img_disp1 = np.concatenate([img1, img2], axis=1)
    cv2.namedWindow("img1, img2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img1, img2", 2 * win_size[0], win_size[1])
    cv2.imshow("img1, img2", img_disp1)

    if flow_arrow is not None:
        cv2.namedWindow("flow_arrow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("flow_arrow", win_size[0], win_size[1])
        cv2.imshow("flow_arrow", flow_arrow)

    if flow_color is not None:
        cv2.namedWindow("flow_color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("flow_color", win_size[0], win_size[1])
        cv2.imshow("flow_color", flow_color)

    cv2.waitKey(300)


"""
General visualization functions
"""


def draw_rectangle(
    center_x,
    center_y,
    size_x,
    size_y,
    ang=0.0,
    edgecolor="dimgray",
    facecolor=None,
    linewidth=2,
):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    offset = np.matmul(R, np.array([[0.5 * size_x], [0.5 * size_y]]))
    anchor_x = center_x - offset[0]
    anchor_y = center_y - offset[1]
    rect = Rectangle(
        (anchor_x, anchor_y),
        size_x,
        size_y,
        angle=(np.rad2deg(ang)),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    plt.gca().add_patch(rect)


def draw_circle(center_x, center_y, radius):
    circle = Circle((center_x, center_y), color="dimgray", radius=radius)
    plt.gca().add_patch(circle)


def visualize_imgs(fig, axs, img_list, titles=None, cmap=None):
    for idx, img in enumerate(img_list):

        if img is None:
            continue

        im = axs[idx].imshow(img, cmap=cmap)
        if cmap is not None:
            fig.colorbar(im, ax=axs[idx])
        if titles is not None:
            axs[idx].set_title(titles[idx])
