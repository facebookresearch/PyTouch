# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import open3d as o3d
import torch

_log = logging.getLogger(__name__)


@dataclass
class Visualizer3DWindowParams:
    top: int = 0
    left: int = 750
    width: int = 1080
    height: int = 1080


@dataclass
class Visualizer3DViewParams:
    fov: int = 0
    front: List[float] = field(default_factory=lambda: [0.4257, -0.2125, -0.8795])
    lookat: List[float] = field(default_factory=lambda: [0.02, 0.0, 0.0])
    up: List[float] = field(default_factory=lambda: [0.9768, -0.0694, 0.2024])
    zoom: float = 0.25


@dataclass
class Visualizer3DOptParams:
    show_coordinate_frame: bool = True
    background_color: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.6])


class Visualizer3D:
    def __init__(
        self,
        base_path: str = "",
        sensor_mesh_file: str = None,
        window_params: Optional[Visualizer3DWindowParams] = None,
        view_params: Optional[Visualizer3DViewParams] = None,
        opt_params: Optional[Visualizer3DOptParams] = None,
        show_sensor_mesh: bool = False,
        t_sleep: float = 0.1,
    ):
        self.t_sleep = t_sleep
        self.base_path = base_path
        self.sensor_mesh_file = sensor_mesh_file
        self.view_params = (
            view_params if view_params is not None else Visualizer3DViewParams()
        )
        self.window_params = (
            window_params if window_params is not None else Visualizer3DWindowParams()
        )
        self.opt_params = (
            opt_params if opt_params is not None else Visualizer3DOptParams()
        )

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            top=self.window_params.top,
            left=self.window_params.left,
            width=self.window_params.width,
            height=self.window_params.height,
        )

        self.set_view()

        self.sensor_mesh = None
        if show_sensor_mesh and sensor_mesh_file is not None:
            self.sensor_mesh = self.create_sensor_mesh(self.sensor_mesh_file)

    def _init_geom_cloud(self):
        return o3d.geometry.PointCloud()

    def _init_geom_frame(self, frame_size=0.01, frame_origin=[0, 0, 0]):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=frame_origin
        )

    def _init_geom_mesh(self, mesh_name, color=None, wireframe=False):
        model_path = os.path.join(self.base_path, mesh_name)
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()

        if wireframe:
            mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        if color is not None:
            mesh.paint_uniform_color(color)
        return mesh

    def init_geometry(
        self,
        geom_type,
        num_items=1,
        sizes=None,
        file_names=None,
        colors=None,
        wireframes=None,
    ):
        geom_list = []
        for i in range(0, num_items):

            if geom_type == "cloud":
                geom = self._init_geom_cloud()
            elif geom_type == "frame":
                frame_size = sizes[i] if sizes is not None else 0.001
                geom = self._init_geom_frame(frame_size=frame_size)
            elif geom_type == "mesh":
                color = colors[i] if colors is not None else None
                wireframe = wireframes[i] if wireframes is not None else False
                geom = self._init_geom_mesh(file_names[i], color, wireframe)
            else:
                _log.error(f"Geometry type (geom_type) error: {geom_type} not found.")
                raise ValueError(
                    "An invalid geom_type was specified, must be one of cloud, frame, mesh"
                )
            geom_list.append(geom)
        return geom_list

    def create_sensor_mesh(
        self, mesh_file, geom_type="mesh", colors=[0.55, 0.55, 0.55]
    ):
        meshes = self.init_geometry(
            geom_type=geom_type,
            num_items=1,
            colors=[colors],
            file_names=[mesh_file],
            wireframes=[True],
        )
        return meshes

    def set_view(self, extrinsics=None):
        ctr = self.vis.get_view_control()
        if extrinsics is not None:
            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = extrinsics
            ctr.convert_from_pinhole_camera_parameters(cam)
        else:
            ctr.change_field_of_view(self.view_params.fov)
            ctr.set_front(self.view_params.front)
            ctr.set_lookat(self.view_params.lookat)
            ctr.set_up(self.view_params.up)
            ctr.set_zoom(self.view_params.zoom)

    def transform_geometry_absolute(self, transform_list, geom_list):
        for idx, geom in enumerate(geom_list):
            T = transform_list[idx]
            geom.transform(T)

    def transform_geometry_relative(
        self, transform_prev_list, transform_curr_list, geom_list
    ):
        for idx, geom in enumerate(geom_list):
            T_prev = transform_prev_list[idx]
            T_curr = transform_curr_list[idx]

            # a. rotate R1^{-1}*R2 about center t1
            geom.rotate(
                torch.matmul(torch.inverse(T_prev[0:3, 0:3]), T_curr[0:3, 0:3]),
                center=(T_prev[0, -1], T_prev[1, -1], T_prev[2, -1]),
            )

            # b. translate by t2 - t1
            geom.translate(
                (
                    T_curr[0, -1] - T_prev[0, -1],
                    T_curr[1, -1] - T_prev[1, -1],
                    T_curr[2, -1] - T_prev[2, -1],
                )
            )

    def add_geometry(self, geom_list):
        if geom_list is None:
            return
        for geom in geom_list:
            self.vis.add_geometry(geom)

    def remove_geometry(self, geom_list, reset_bounding_box=False):
        if geom_list is None:
            return
        for geom in geom_list:
            self.vis.remove_geometry(geom, reset_bounding_box=reset_bounding_box)

    def update_geometry(self, geom_list):
        for geom in geom_list:
            self.vis.update_geometry(geom)

    def clear_geometries(self):
        self.vis.clear_geometries()

    @staticmethod
    def visualize_inlier_outlier(cloud, idx):
        inlier_cloud = cloud.select_by_index(idx)
        outlier_cloud = cloud.select_by_index(idx, invert=True)

        _log.info("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries(
            [inlier_cloud, outlier_cloud],
            zoom=0.3412,
            front=[0.4257, -0.2125, -0.8795],
            lookat=[2.6172, 2.0475, 1.532],
            up=[-0.0694, -0.9768, 0.2024],
        )

    def visualize_geometries(
        self, clouds=None, frames=None, meshes=None, transforms=None, extrinsics=None
    ):
        if meshes is not None:
            meshes = [copy.deepcopy(mesh) for mesh in meshes]
            if transforms is not None:
                self.transform_geometry_absolute(transforms, meshes)

        if frames is not None:
            frames = [copy.deepcopy(frame) for frame in frames]
            if transforms is not None:
                self.transform_geometry_absolute(transforms, frames)

        ctr = self.vis.get_view_control()
        camera_view = ctr.convert_to_pinhole_camera_parameters()

        self.add_geometry(clouds)
        self.add_geometry(meshes)
        self.add_geometry(frames)

        ctr.convert_from_pinhole_camera_parameters(camera_view)
        self._render(extrinsics=extrinsics)

        self.remove_geometry(clouds)
        self.remove_geometry(meshes)
        self.remove_geometry(frames)

    def points_to_cloud(self, points_3d, colors=None):
        clouds = [o3d.geometry.PointCloud()]
        for idx, cloud in enumerate(clouds):
            points_3d_np = (
                points_3d[idx].to("cpu").detach().numpy()
                if torch.is_tensor(points_3d[idx])
                else points_3d[idx]
            )
        cloud.points = copy.deepcopy(
            o3d.utility.Vector3dVector(points_3d_np.transpose())
        )
        if colors is not None:
            cloud.paint_uniform_color(colors[idx])
        return clouds

    def _render(self, extrinsics=None):
        # todo(lambetam) refactor set view with extrinsics self.set_view(extrinsics)
        self.vis.poll_events()
        self.vis.update_renderer()

    def render(self, points_3d, extrinsics=None):
        clouds = self.points_to_cloud(points_3d=[points_3d])
        self.visualize_geometries(
            clouds=clouds, meshes=self.sensor_mesh, extrinsics=extrinsics
        )

    def destroy(self):
        self.vis.destroy_window()
