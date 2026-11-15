"""Kinematics for the SO-101 arm.

This package provides three things:

- ``transforms.py``: numpy SO(3)/SE(3) transform utilities.
- ``dh.py``: standard (classical) Denavit-Hartenberg forward kinematics of the
  five-joint arm, with the ``gripperframe`` site as the end effector, checked
  against the vendored MuJoCo model in the tests.
- ``camera_frames.py``: the MuJoCo camera-frame to OpenCV/optical-frame pose
  conversion. It is numpy only and does not import ``mujoco``.

Arm forward kinematics is implemented. Inverse kinematics is not implemented
yet (later phase).
"""

from .camera_frames import MUJOCO_TO_OPENCV_ROTATION, mujoco_camera_to_opencv
from .dh import (
    ARM_JOINT_NAMES,
    SO101_DH_PARAMS,
    SO101_T_BASE_TO_0,
    SO101_T_N_TO_EE,
    dh_transform,
    forward_kinematics,
)
from .transforms import (
    axis_angle_to_rotation,
    compose_quaternion_body,
    compose_quaternion_fixed,
    compose_se3_body,
    compose_se3_fixed,
    compose_so3_body,
    compose_so3_fixed,
    identity_se3,
    identity_so3,
    inverse_quaternion,
    inverse_se3,
    inverse_so3,
    normalize_quaternion,
    pose,
    quaternion_to_rotation,
    rotation_of,
    rotation_to_axis_angle,
    rotation_to_quaternion,
    rotation_to_rpy,
    rotation_to_zyz,
    rpy_to_rotation,
    rx,
    ry,
    rz,
    transform_point,
    translation_of,
    zyz_to_rotation,
)

__all__ = [
    "ARM_JOINT_NAMES",
    "MUJOCO_TO_OPENCV_ROTATION",
    "SO101_DH_PARAMS",
    "SO101_T_BASE_TO_0",
    "SO101_T_N_TO_EE",
    "axis_angle_to_rotation",
    "compose_quaternion_body",
    "compose_quaternion_fixed",
    "compose_se3_body",
    "compose_se3_fixed",
    "compose_so3_body",
    "compose_so3_fixed",
    "dh_transform",
    "forward_kinematics",
    "identity_se3",
    "identity_so3",
    "inverse_quaternion",
    "inverse_se3",
    "inverse_so3",
    "mujoco_camera_to_opencv",
    "normalize_quaternion",
    "pose",
    "quaternion_to_rotation",
    "rotation_of",
    "rotation_to_axis_angle",
    "rotation_to_quaternion",
    "rotation_to_rpy",
    "rotation_to_zyz",
    "rpy_to_rotation",
    "rx",
    "ry",
    "rz",
    "transform_point",
    "translation_of",
    "zyz_to_rotation",
]
