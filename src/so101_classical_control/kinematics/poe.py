"""Product-of-exponentials kinematics for the SO-101 arm, derived from the DH model.

The home configuration ``M`` and the space-frame screws ``S`` are computed from
the frozen standard-DH constants in :mod:`~so101_classical_control.kinematics.dh`
at ``q = 0``. No ``mujoco`` import is involved: the numbers come from the same
in-repo DH table that the forward kinematics uses.

For each revolute joint the screw is the joint axis at home, expressed in the
base frame::

    omega_i = R_{0,i-1}(0) @ [0, 0, 1]
    v_i     = -omega_i x q_{i-1}(0)

with ``q_{i-1}(0)`` the origin of DH frame ``i - 1`` at home. The forward
kinematics is then MR's product of exponentials,
``T(θ) = exp([S1] θ1) ... exp([S5] θ5) M``, which the existing
``dh.forward_kinematics`` must reproduce to tight tolerance (P3).

``modern_robotics`` is imported as a package; its source is not vendored.
"""

from __future__ import annotations

import numpy as np
from modern_robotics import (
    Adjoint as _mr_adjoint,
)
from modern_robotics import (
    FKinSpace as _mr_fk_in_space,
)
from modern_robotics import (
    JacobianBody as _mr_jacobian_body,
)
from modern_robotics import (
    JacobianSpace as _mr_jacobian_space,
)

from .dh import (
    ARM_JOINT_NAMES,
    SO101_DH_PARAMS,
    SO101_T_BASE_TO_0,
    arm_joint_angles,
    dh_transform,
    forward_kinematics,
)
from .transforms import compose_se3_body

__all__ = [
    "SO101_BLIST",
    "SO101_M",
    "SO101_SLIST",
    "adjoint",
    "body_jacobian",
    "fk_in_space",
    "home_pose",
    "space_jacobian",
]

_DTYPE = np.float64


def _home_pose() -> np.ndarray:
    """Return ``gripperframe`` at ``q = 0`` from the DH chain."""
    return forward_kinematics(np.zeros(len(ARM_JOINT_NAMES), dtype=_DTYPE))


def _space_screws() -> np.ndarray:
    """Return the ``(6, 5)`` space-frame screws evaluated at ``q = 0``."""
    transform = SO101_T_BASE_TO_0.copy()
    screws = np.zeros((6, len(ARM_JOINT_NAMES)), dtype=_DTYPE)
    for index, (a, alpha, d, theta_offset) in enumerate(SO101_DH_PARAMS):
        rotation = transform[:3, :3]
        origin = transform[:3, 3]
        omega = rotation @ np.array([0.0, 0.0, 1.0], dtype=_DTYPE)
        screw = np.concatenate([omega, -np.cross(omega, origin)])
        screws[:, index] = screw
        transform = compose_se3_body(
            transform, dh_transform(a, alpha, d, theta_offset)
        )
    return screws


def _body_screws() -> np.ndarray:
    """Return the ``(6, 5)`` body-frame screws ``Adjoint(inv(M)) S``."""
    return _mr_adjoint(np.linalg.inv(SO101_M)) @ SO101_SLIST


#: Home configuration ``M = forward_kinematics(zeros)``, ``(4, 4)``.
SO101_M: np.ndarray = _home_pose()

#: Space-frame screws in :data:`~so101_classical_control.kinematics.ARM_JOINT_NAMES`
#: order, ``(6, 5)``, evaluated at ``q = 0``.
SO101_SLIST: np.ndarray = _space_screws()

#: Body-frame screws ``Adjoint(inv(M)) Slist``, ``(6, 5)``.
SO101_BLIST: np.ndarray = _body_screws()


def home_pose() -> np.ndarray:
    """Return a copy of the SO-101 home pose ``M`` (``gripperframe`` at ``q = 0``)."""
    return SO101_M.copy()


def adjoint(transform: np.ndarray) -> np.ndarray:
    """Return the ``(6, 6)`` adjoint ``Ad_T`` of an SE(3) transform.

    The adjoint maps a body twist to the same twist expressed in the space
    frame, ``V_s = Ad_T V_b``.

    Raises
    ------
    ValueError
        If ``transform`` is not a valid ``(4, 4)`` SE(3) transform.
    """
    array = np.asarray(transform, dtype=_DTYPE)
    if array.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {array.shape}")
    # ``compose_se3_body`` validates the last row, orthonormality, and determinant.
    compose_se3_body(array, np.eye(4, dtype=_DTYPE))
    return _mr_adjoint(array)


def fk_in_space(joint_angles) -> np.ndarray:
    """Return ``FKinSpace(M, Slist, θ)``, the ``gripperframe`` pose in the base frame.

    ``joint_angles`` accepts the same forms as
    :func:`~so101_classical_control.kinematics.dh.forward_kinematics`: five arm
    angles, six with a trailing gripper value that is ignored, or a mapping keyed
    by joint name.

    Raises
    ------
    ValueError
        If ``joint_angles`` is malformed, non-finite, or names unknown joints.
    """
    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    return np.asarray(_mr_fk_in_space(SO101_M, SO101_SLIST, angles), dtype=_DTYPE)


def space_jacobian(joint_angles) -> np.ndarray:
    """Return ``JacobianSpace(Slist, θ)``, the ``(6, 5)`` space Jacobian.

    The six rows are the spatial angular and linear velocity components; the
    five columns are the five arm joints. The gripper DoF is excluded.

    Raises
    ------
    ValueError
        If ``joint_angles`` is malformed, non-finite, or names unknown joints.
    """
    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    return np.asarray(_mr_jacobian_space(SO101_SLIST, angles), dtype=_DTYPE)


def body_jacobian(joint_angles) -> np.ndarray:
    """Return ``JacobianBody(Blist, θ)``, the ``(6, 5)`` body Jacobian.

    It is related to :func:`space_jacobian` by ``J_s = Ad_{T_sb} J_b`` with
    ``T_sb = fk_in_space(θ)``.

    Raises
    ------
    ValueError
        If ``joint_angles`` is malformed, non-finite, or names unknown joints.
    """
    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    return np.asarray(_mr_jacobian_body(SO101_BLIST, angles), dtype=_DTYPE)
