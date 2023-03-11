# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from eg3d.training.volumetric_rendering import math_utils


class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu', sample_mode='randn'):
        if sample_mode == 'randn':
            h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
            v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        else:
            # Uniform sample to sample diverse camera position
            h = torch.rand((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
            v = torch.rand((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def sample_camera(batch_size=1, yaw_range=0.35, pitch_range=0.25, device='cpu'):
    angle_p = -0.2
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    extrinsics = LookAtPoseSampler.sample(horizontal_mean=np.pi/2, vertical_mean=np.pi/2 + angle_p, lookat_position=camera_lookat_point, 
                        horizontal_stddev=yaw_range, vertical_stddev=pitch_range, radius=2.7, batch_size=batch_size, device=device, sample_mode='uniform')
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).view(1, 9).repeat(batch_size, 1)
    camera = torch.cat([extrinsics.view(-1, 16), intrinsics.view(-1, 9)], dim=1)
    return camera


def angle_to_rotation(yaw, pitch, roll=0):
    # yaw -> roll
    # pitch -> yaw
    # roll -> pitch
    rollMatrix = np.matrix([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    yawMatrix = np.matrix([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    pitchMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = torch.from_numpy(R)
    return R


def sample_surrounding_camera(middle_camera, batch_size=1, yaw_range=0.1, pitch_range=0.1):
    device = middle_camera.device
    y = (torch.rand((batch_size, 1), device=device) * 2 - 1) * yaw_range + 0.0
    p = (torch.rand((batch_size, 1), device=device) * 2 - 1) * pitch_range + 0.0
    rot_list = []
    for _y, _p in zip(y, p):
        rot_list.append(angle_to_rotation(yaw=_y, pitch=_p, roll=0))
    sample_rotation = torch.stack(rot_list, dim=0).view(batch_size, 3, 3).float().to(device)
    
    middle_camera = middle_camera.repeat(batch_size, 1)
    middle_extrinsics = middle_camera[:, :16].view(-1, 4, 4)
    
    middle_extrinsics[:, :3] = torch.bmm(sample_rotation, middle_extrinsics[:, :3])
    new_middle_camera = middle_camera.clone()
    new_middle_camera[:, :16] = middle_extrinsics.view(-1, 16)
    return new_middle_camera



def calculate_surrounding_camera(middle_camera, batch_size=1, yaw_range=0.1, pitch_range=0.1):
    device = middle_camera.device
    y = (torch.ones((batch_size, 1), device=device) * 2 - 1) * yaw_range + 0.0
    p = (torch.ones((batch_size, 1), device=device) * 2 - 1) * pitch_range + 0.0
    rot_list = []
    for _y, _p in zip(y, p):
        rot_list.append(angle_to_rotation(yaw=_y, pitch=_p, roll=0))
    sample_rotation = torch.stack(rot_list, dim=0).view(batch_size, 3, 3).float().to(device)

    middle_camera = middle_camera.repeat(batch_size, 1)
    middle_extrinsics = middle_camera[:, :16].view(-1, 4, 4)
    
    middle_extrinsics[:, :3] = torch.bmm(sample_rotation, middle_extrinsics[:, :3])
    new_middle_camera = middle_camera.clone()
    new_middle_camera[:, :16] = middle_extrinsics.view(-1, 16)
    return new_middle_camera


def cal_canonical_c(yaw_angle=0, pitch_angle=0, batch_size=1, device='cpu'):
    angle_p = -0.2
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    
    extrinsics = LookAtPoseSampler.sample(np.pi/2 + yaw_angle, np.pi/2 + angle_p + pitch_angle, camera_lookat_point, radius=2.7, batch_size=batch_size, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).view(1, 9).repeat(batch_size, 1)
    camera = torch.cat([extrinsics.view(-1, 16), intrinsics], dim=1)
    return camera


try:
    CANONICAL_CAMERA = cal_canonical_c(0, 0, 1, 'cuda')
    CANONICAL_ROTATION = CANONICAL_CAMERA.view(-1, 25)[:, :16].view(-1, 4, 4)[:, :3, :3]
except:
    CANONICAL_CAMERA = None
    CANONICAL_ROTATION = None


def cal_sequence_c():
    sequence_c = []
    for _i in range(0, 15, 1):
        rg = _i * 0.1 - 0.7
        c = cal_canonical_c(rg, 0, 1, device='cpu')
        sequence_c.append(c)
    
    sequence_c = torch.cat(sequence_c, dim=0)
    return sequence_c


def cal_sequence_c_2():
    sequence_c = []
    c = cal_canonical_c(-0.65, 0, 1, device='cpu')
    sequence_c.append(c) 
    c = cal_canonical_c(0.65, 0, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(-0.4, 0.2, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0.4, -0.2, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(-0.2, -0.2, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0.2, 0.2, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0, 0, 1, device='cpu')
    sequence_c.append(c)
    sequence_c = torch.cat(sequence_c, dim=0)
    return sequence_c


def cal_sequence_c_3(camera):
    sequence_c = []
    c = calculate_surrounding_camera(camera, 1, 0.1, 0.1)
    sequence_c.append(c) 
    c = calculate_surrounding_camera(camera, 1, -0.3, 0.2)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.35, 0.1)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.3, 0)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.3, -0.1)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.3, -0.2)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.2, -0.2)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.1, 0.3)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.1, 0.1)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.1, 0.4)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, 0.1, -0.3)
    sequence_c.append(c)
    c = calculate_surrounding_camera(camera, 1, -0.3, 0.3)
    sequence_c.append(c)
    sequence_c = torch.cat(sequence_c, dim=0)
    return sequence_c


def cal_sequence_c_4():
    sequence_c = []
    c = cal_canonical_c(-0.4, 0.3, 1, device='cpu')
    sequence_c.append(c) 
    c = cal_canonical_c(-0.4, 0, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(-0.4, -0.3, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0, 0.3, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0, 0, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0, -0.3, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0.4, 0.3, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0.4, 0, 1, device='cpu')
    sequence_c.append(c)
    c = cal_canonical_c(0.4, -0.3, 1, device='cpu')
    sequence_c.append(c)
    sequence_c = torch.cat(sequence_c, dim=0)
    return sequence_c


def flip_yaw(pose_matrix):
    flipped = pose_matrix.clone()
    flipped[:, 0, 1] *= -1
    flipped[:, 0, 2] *= -1
    flipped[:, 0, 3] *= -1
    flipped[:, 1, 0] *= -1
    flipped[:, 2, 0] *= -1
    return flipped


def cal_mirror_c(camera):
    pose, intrinsics = camera[:, :16].reshape(-1, 4, 4), camera[:, 16:].reshape(-1, 3, 3)
    flipped_pose = flip_yaw(pose)
    mirror_camera = torch.cat([flipped_pose.view(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    return mirror_camera


def rotation_to_angle(matrix):
    # assert camera.shape[0] == 1
    # matrix = camera
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    pitch = torch.arctan(-r23 / r33)
    yaw = torch.arctan(r13 * torch.cos(pitch) / r33)
    roll = torch.arctan(-r12 / r11)
    # print(theta1, theta2, theta3)
    return yaw, pitch, roll


# linear weight
def cal_camera_weight_linear(camera):
    weight = []
    for c in camera:
        y, p, r = rotation_to_angle(c.view(25)[:16].view(4, 4)[:3, :3])
        # torch.relu() TanH or SigMoid
        w = min(torch.abs(y), 1)
        if w < 0.2:
            w = torch.zeros_like(w)
        weight.append(w)
    weight = torch.stack(weight, dim=0)
    return weight

try:
    GAUSS_CONST = torch.sqrt(torch.tensor(2 * torch.pi)).cuda()
except:
    GAUSS_CONST = None

def gauss_function(x, mean=0.0, std=0.25):
    f = torch.exp(-0.5 * (x - mean) * (x - mean) / std / std) / (std * GAUSS_CONST)
    return f

def cal_camera_gauss_weight(camera):
    weight = []
    for c in camera:
        y, p, r = rotation_to_angle(c.view(25)[:16].view(4, 4)[:3, :3])
        w = gauss_function(y, std=0.4)/2.6
        weight.append(w)
    return weight


def cal_camera_weight(camera):
    weight = []
    for c in camera:
        y, p, r = rotation_to_angle(c.view(25)[:16].view(4, 4)[:3, :3])
        y = torch.abs(y)
        # torch.relu() TanH or SigMoid
        w = gauss_function(y, std=0.29)/2.7
        w = (1 - w) / 2
        # w = min(w, 1)
        if y < 0.2:
            w = torch.zeros_like(w)
        weight.append(w)
    weight = torch.stack(weight, dim=0)
    return weight


def rotationMatrixToEulerAngles(R):
    sy = torch.sqrt(R[:, 0,0] * R[:, 0,0] +  R[:, 1,0] * R[:, 1, 0])
    # singular = sy < 1e-6
    # if not singular :
    x = torch.atan2(R[:, 2,1] , R[:, 2,2])  # x = math.atan2(R[2,1] , R[2,2])
    y = torch.atan2(-R[:, 2,0], sy)
    z = torch.atan2(R[:, 1,0], R[:, 0,0])
    return x, y, z



def check_front(camera, EPS=0.1):
    rotation = camera.view(-1, 25)[:, :16].view(-1, 4, 4)[:, :3, :3]
    x, y, z= rotationMatrixToEulerAngles(rotation)
    if_front = (torch.abs(y) < EPS) # * (torch.abs(x) - 3.0037 < 0.005)
    return if_front
