"""Kinematics for the SO-101 arm.

This package provides these means modules:

- ``transforms.py``: numpy SO(3)/SE(3) transform utilities.
- ``dh.py``: standard (classical) Denavit-Hartenberg forward kinematics of the
  five-joint arm, with the ``gripperframe`` site as the end effector, checked
  against the vendored MuJoCo model in the tests.
- ``poe.py``: product-of-exponentials data for the same arm, derived from the DH
  model at ``q = 0``, plus thin wrappers over ``modern_robotics``
  (``FKinSpace``, ``JacobianSpace``, ``JacobianBody``, ``Adjoint``).
- ``planar.py``: planar two-link (2R) closed-form kinematics, analytical IK,
  reachability, and manipulability.
- ``inverse.py``: damped least-squares / pseudo-inverse inverse kinematics and
  null-space projection built on the Jacobian means.
- ``camera_frames.py``: the MuJoCo camera-frame to OpenCV/optical-frame pose
  conversion. It is numpy only and does not import ``mujoco``.

Forward kinematics is implemented in both DH and PoE form. Inverse kinematics is
implemented numerically in ``inverse.py`` for the five arm joints and in closed
form for the planar 2R arm. The goal-level wrappers live in
``so101_classical_control.goals``.
"""

from .camera_frames import MUJOCO_TO_OPENCV_ROTATION, mujoco_camera_to_opencv
from .dh import (
    ARM_JOINT_NAMES,
    SO101_DH_PARAMS,
    SO101_T_BASE_TO_0,
    SO101_T_N_TO_EE,
    arm_joint_angles,
    dh_transform,
    forward_kinematics,
)
from .inverse import (
    IK_DAMPING,
    IK_MAX_ITERATIONS,
    IK_MAX_STEP,
    IK_ROTATION_TOL,
    IK_TRANSLATION_TOL,
    IKResult,
    damped_pseudo_inverse,
    null_space_projector,
    pose_error,
    position_jacobian,
    solve_pose_ik,
    solve_position_ik,
)
from .planar import (
    PLANAR_BOUNDARY_GUARD,
    planar_forward_kinematics,
    planar_ik_solutions,
    planar_inverse_kinematics,
    planar_jacobian,
    planar_manipulability,
    planar_reachable,
)
from .poe import (
    SO101_BLIST,
    SO101_M,
    SO101_SLIST,
    adjoint,
    body_jacobian,
    fk_in_space,
    space_jacobian,
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
    "IK_DAMPING",
    "IK_MAX_ITERATIONS",
    "IK_MAX_STEP",
    "IK_ROTATION_TOL",
    "IK_TRANSLATION_TOL",
    "IKResult",
    "MUJOCO_TO_OPENCV_ROTATION",
    "PLANAR_BOUNDARY_GUARD",
    "SO101_BLIST",
    "SO101_DH_PARAMS",
    "SO101_M",
    "SO101_SLIST",
    "SO101_T_BASE_TO_0",
    "SO101_T_N_TO_EE",
    "adjoint",
    "arm_joint_angles",
    "axis_angle_to_rotation",
    "body_jacobian",
    "compose_quaternion_body",
    "compose_quaternion_fixed",
    "compose_se3_body",
    "compose_se3_fixed",
    "compose_so3_body",
    "compose_so3_fixed",
    "damped_pseudo_inverse",
    "dh_transform",
    "fk_in_space",
    "forward_kinematics",
    "identity_se3",
    "identity_so3",
    "inverse_quaternion",
    "inverse_se3",
    "inverse_so3",
    "mujoco_camera_to_opencv",
    "normalize_quaternion",
    "null_space_projector",
    "planar_forward_kinematics",
    "planar_inverse_kinematics",
    "planar_ik_solutions",
    "planar_jacobian",
    "planar_manipulability",
    "planar_reachable",
    "pose",
    "pose_error",
    "position_jacobian",
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
    "solve_pose_ik",
    "solve_position_ik",
    "space_jacobian",
    "transform_point",
    "translation_of",
    "zyz_to_rotation",
]
