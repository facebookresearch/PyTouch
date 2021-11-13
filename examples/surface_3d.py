# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import hydra

from pytouch.common.visualizer import Visualizer3D, Visualizer3DViewParams
from pytouch.datasets.sequence import ImageSequenceDataset
from pytouch.sensors import DigitSensor
from pytouch.tasks import Surface3D


@hydra.main(config_path="configs", config_name="digit_surface3d.yaml")
def visualize_surface_3d(cfg):
    # load touch sequence dataset
    img_seq_ds = ImageSequenceDataset(cfg.dataset.path)

    # define a custom camera view
    view_params = Visualizer3DViewParams(
        fov=60,  # field of view
        front=[0.4257, -0.2125, -0.8795],  # front vector
        lookat=[0.02, 0.0, 0.0],  # look at vector
        up=[0.9768, -0.0694, 0.2024],  # up vector
        zoom=0.25,  # zoom
    )

    # initialize point cloud visualizer
    visualizer = Visualizer3D(view_params=view_params)

    # initialize surface 3d model
    surface3d = Surface3D(
        DigitSensor,
        sensor_params=cfg.sensor,
    )

    # get first sequence
    sequence = img_seq_ds[0]
    for img in sequence:
        output = surface3d.point_cloud_3d(img_color=img)
        visualizer.render(output.points_3d)
        # you may also plot the color, predicted normal, and predicted depth images
        # color = output.color
        # depth = output.depth
        # normal = output.normal


if __name__ == "__main__":
    visualize_surface_3d()
