# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import logging
from dataclasses import dataclass

import numpy.typing as npt
import torch
import torch.nn as nn

import pytouch.tasks.surface3d.geometry as geometry
from pytouch.models import Pix2PixModel, PyTouchZoo

_log = logging.getLogger(__name__)


@dataclass
class Surface3DModelDefaults:
    name: str = "sim_sphere_sim_cube_real_sphere"
    model_type: str = "pix2pix"
    dataset_mode: str = "aligned"
    checkpoints_dir: str = "none"
    direction: str = "AtoB"


class Surface3D(nn.Module):
    @dataclass
    class Surface3DReturn:
        points_3d: npt.NDArray
        color: npt.NDArray
        normal: npt.NDArray
        depth: npt.NDArray

    def __init__(
        self,
        sensor,
        sensor_params,
        model_params=Surface3DModelDefaults,
        zoo_model="p2p_surface_3d",
        model_path="",
    ):
        super(Surface3D, self).__init__()
        self.sensor = sensor
        self.sensor_params = sensor_params
        self.model_path = model_path
        self.model = Pix2PixModel(
            model_params.name,
            model_params.model_type,
            model_dir=model_path,
            dataset_mode=model_params.dataset_mode,
            direction=model_params.direction,
        )
        if not model_path:
            zoo = PyTouchZoo()
            state_dict = zoo.load_model_from_zoo(zoo_model, sensor)
            self.model.init_zoo_model(state_dict)

    def __call__(self):
        return self.point_cloud_3d()

    def normals(self, img_input):
        img_normal_pred = self.model.color_to_normal(img_input).to("cpu")
        # todo(lambetam) determine if sensor or sim
        return img_normal_pred

    def normal_to_grad_depth(self, img_normal, gel_width, gel_height, bg_mask):
        img_normal = geometry.preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
        grad_x, grad_y = geometry.normal_to_grad_depth(
            img_normal=img_normal,
            gel_width=gel_width,
            gel_height=gel_height,
            bg_mask=bg_mask,
        )
        return img_normal, grad_x, grad_y

    def grad_depth_to_depth(
        self,
        img_normal,
        grad_x,
        grad_y,
        bg_mask,
        remove_bg_depth=True,
        depth=0.02,
        max_depth=None,
    ):
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
        img_depth = geometry.integrate_grad_depth(
            grad_x, grad_y, boundary=boundary, bg_mask=bg_mask, max_depth=depth
        )
        if remove_bg_depth:
            img_depth = geometry.mask_background(
                img_depth, bg_mask=(img_depth >= max_depth), bg_val=0.0
            )
        return img_depth

    def depth_to_points3d(self, img_depth, view_mat, proj_mat):
        view_mat = (
            view_mat if isinstance(view_mat, torch.Tensor) else torch.tensor(view_mat)
        )
        proj_mat = (
            view_mat if isinstance(proj_mat, torch.Tensor) else torch.tensor(proj_mat)
        )

        view_mat = torch.inverse(view_mat)

        points_3d = geometry.depth_to_pts3d(
            depth=img_depth, P=proj_mat, V=view_mat, params=self.sensor_params
        )
        points_3d = geometry.remove_outlier_pts(
            points_3d, nb_neighbors=20, std_ratio=10.0
        )
        return points_3d

    def point_cloud_3d(self, img_color, img_normal_gt=None, img_depth_gt=None):
        normal_pred = self.normals(img_color)

        color = copy.deepcopy(img_color)
        normal = copy.deepcopy(normal_pred)

        # TODO (psodhi): Background gt normals nx, ny are non-zero for sim but correctly zero for real.
        # Hence, we mask out background in sim relying on gt depth. Once background is fixed,
        # we can remove the code snippet below.
        bg_mask = None
        if self.sensor == "sim":
            depth_gt = copy.deepcopy(img_depth_gt)
            bg_mask = (depth_gt > self.gel_depth).squeeze()

        img_grad_depth, grad_x, grad_y = self.normal_to_grad_depth(
            normal, self.sensor_params.gel_width, self.sensor_params.gel_height, bg_mask
        )

        img_depth = self.grad_depth_to_depth(
            img_grad_depth,
            grad_x,
            grad_y,
            bg_mask,
            remove_bg_depth=self.sensor_params.remove_background_depth,
            max_depth=self.sensor_params.max_depth,
        )

        img_points3d = self.depth_to_points3d(
            img_depth, self.sensor_params.T_cam_offset_sim, self.sensor_params.P
        )

        # numpy
        img_color_np = (color.permute(1, 2, 0)).cpu().detach().numpy()
        img_normal_np = (normal.permute(1, 2, 0)).cpu().detach().numpy()
        img_depth_np = img_depth.cpu().detach().numpy()

        return self.Surface3DReturn(
            img_points3d, img_color_np, img_normal_np, img_depth_np
        )
