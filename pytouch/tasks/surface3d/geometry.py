# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy

import numpy as np
import open3d as o3d
import torch

from pytouch.models.pix2pix.thirdparty import poisson
from pytouch.utils.common_utils import max_clip


def mask_background(x, bg_mask, bg_val=0.0):
    if bg_mask is not None:
        x[bg_mask] = bg_val
    return x


def preproc_normal(img_normal, bg_mask=None):
    """
    img_normal: lies in range [0, 1]
    """

    # 0.5 corresponds to 0
    img_normal = img_normal - 0.5

    # normalize
    img_normal = img_normal / torch.linalg.norm(img_normal, dim=0)

    # set background to have only z normals (flat, facing camera)
    if bg_mask is not None:
        img_normal[0:2, bg_mask] = 0.0
        img_normal[2, bg_mask] = 1.0

    return img_normal


def normal_to_grad_depth(img_normal, gel_width=1.0, gel_height=1.0, bg_mask=None):
    # Ref: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc/34644939#34644939 # noqa: E501

    EPS = 1e-1
    nz = torch.max(torch.tensor(EPS), img_normal[2, :])

    dzdx = -(img_normal[0, :] / nz).squeeze()
    dzdy = -(img_normal[1, :] / nz).squeeze()

    # taking out negative sign as we are computing gradient of depth not z
    # since z is pointed towards sensor, increase in z corresponds to decrease in depth
    # i.e., dz/dx = -ddepth/dx
    ddepthdx = -dzdx
    ddepthdy = -dzdy

    # sim: pixel axis v points in opposite dxn of camera axis y
    ddepthdu = ddepthdx
    ddepthdv = -ddepthdy

    gradx = ddepthdu  # cols
    grady = ddepthdv  # rows

    # convert units from pixel to meters
    C, H, W = img_normal.shape
    gradx = gradx * (gel_width / W)
    grady = grady * (gel_height / H)

    if bg_mask is not None:
        gradx = mask_background(gradx, bg_mask=bg_mask, bg_val=0.0)
        grady = mask_background(grady, bg_mask=bg_mask, bg_val=0.0)

    return gradx, grady


def integrate_grad_depth(gradx, grady, boundary=None, bg_mask=None, max_depth=0.0):
    if boundary is None:
        boundary = torch.zeros((gradx.shape[0], gradx.shape[1]))

    img_depth_recon = poisson.poisson_reconstruct(
        grady.cpu().detach().numpy(),
        gradx.cpu().detach().numpy(),
        boundary.cpu().detach().numpy(),
    )
    img_depth_recon = torch.FloatTensor(img_depth_recon, device=gradx.device)

    if bg_mask is not None:
        img_depth_recon = mask_background(img_depth_recon, bg_mask)

    # after integration, img_depth_recon lies between 0. (bdry) and a -ve val (obj depth)
    # rescale to make max depth as gel depth and obj depth as +ve values
    img_depth_recon = max_clip(img_depth_recon, max_val=torch.tensor(0.0)) + max_depth

    return img_depth_recon


"""
3D-2D projection / conversion functions
OpenGL transforms reference: http://www.songho.ca/opengl/gl_transform.html
"""


def _vectorize_pixel_coords(rows, cols, device=None):
    y_range = torch.arange(rows, device=device)
    x_range = torch.arange(cols, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
    pixel_pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)  # 2 x N

    return pixel_pos


def _clip_to_pixel(clip_pos, img_shape, params):
    H, W = img_shape

    # clip -> ndc coords
    x_ndc = clip_pos[0, :] / clip_pos[3, :]
    y_ndc = clip_pos[1, :] / clip_pos[3, :]
    # z_ndc = clip_pos[2, :] / clip_pos[3, :]

    # ndc -> pixel coords
    x_pix = (W - 1) / 2 * (x_ndc + 1)  # [-1, 1] -> [0, W-1]
    y_pix = (H - 1) / 2 * (y_ndc + 1)  # [-1, 1] -> [0, H-1]
    # z_pix = (f-n) / 2 *  z_ndc + (f+n) / 2

    pixel_pos = torch.stack((x_pix, y_pix), dim=0)
    return pixel_pos


def _pixel_to_clip(pix_pos, depth_map, params):
    """
    :param pix_pos: position in pixel space, (2, N)
    :param depth_map: depth map, (H, W)
    :return: clip_pos position in clip space, (4, N)
    """
    x_pix = pix_pos[0, :]
    y_pix = pix_pos[1, :]

    H, W = depth_map.shape
    f = params.z_far
    n = params.z_near

    # pixel -> ndc coords
    x_ndc = 2 / (W - 1) * x_pix - 1  # [0, W-1] -> [-1, 1]
    y_ndc = 2 / (H - 1) * y_pix - 1  # [0, H-1] -> [-1, 1]
    z_buf = depth_map[y_pix, x_pix]

    # ndc -> clip coords
    z_eye = -z_buf
    w_c = -z_eye
    x_c = x_ndc * w_c
    y_c = y_ndc * w_c
    z_c = -(f + n) / (f - n) * z_eye - 2 * f * n / (f - n) * 1.0

    clip_pos = torch.stack([x_c, y_c, z_c, w_c], dim=0)
    return clip_pos


def _clip_to_eye(clip_pos, P):
    P_inv = torch.inverse(P)
    eye_pos = torch.matmul(P_inv, clip_pos)
    return eye_pos


def _eye_to_clip(eye_pos, P):
    clip_pos = torch.matmul(P, eye_pos)
    return clip_pos


def _eye_to_world(eye_pos, V):
    V_inv = torch.inverse(V)
    world_pos = torch.matmul(V_inv, eye_pos)
    world_pos = world_pos / world_pos[3, :]
    return world_pos


def _world_to_eye(world_pos, V):
    eye_pos = torch.matmul(V, world_pos)
    return eye_pos


def _world_to_object(world_pos, M):
    M_inv = torch.inverse(M)
    obj_pos = torch.matmul(M_inv, world_pos)
    obj_pos = obj_pos / obj_pos[3, :]
    return obj_pos


def _object_to_world(obj_pos, M):
    world_pos = torch.matmul(M, obj_pos)
    world_pos = world_pos / world_pos[3, :]
    return world_pos


def depth_to_pts3d(depth, P, V, params=None, ordered_pts=False):
    """
    :param depth: depth map, (C, H, W) or (H, W)
    :param P: projection matrix, (4, 4)
    :param V: view matrix, (4, 4)
    :return: world_pos position in 3d world coordinates, (3, H, W) or (3, N)
    """
    assert 2 <= len(depth.shape) <= 3
    assert P.shape == (4, 4)
    assert V.shape == (4, 4)

    depth_map = depth.squeeze(0) if (len(depth.shape) == 3) else depth
    H, W = depth_map.shape
    pixel_pos = _vectorize_pixel_coords(rows=H, cols=W)

    clip_pos = _pixel_to_clip(pixel_pos, depth_map, params)
    eye_pos = _clip_to_eye(clip_pos, P)
    world_pos = _eye_to_world(eye_pos, V)

    world_pos = world_pos[0:3, :] / world_pos[3, :]

    if ordered_pts:
        H, W = depth_map.shape
        world_pos = world_pos.reshape(world_pos.shape[0], H, W)

    return world_pos


"""
Open3D helper functions
"""


def remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.0):
    points3d_np = (
        points3d.cpu().detach().numpy() if torch.is_tensor(points3d) else points3d
    )

    cloud = o3d.geometry.PointCloud()
    cloud.points = copy.deepcopy(o3d.utility.Vector3dVector(points3d_np.transpose()))
    cloud_filt, ind_filt = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    points3d_filt = np.asarray(cloud_filt.points).transpose()
    points3d_filt = (
        torch.tensor(points3d_filt) if torch.is_tensor(points3d) else points3d_filt
    )

    return points3d_filt
