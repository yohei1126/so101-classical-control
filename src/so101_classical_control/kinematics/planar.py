"""Planar two-link (2R) arm: closed-form kinematics, IK, and manipulability.

The arm lies in the plane, joint 1 at the origin turning about ``+z``, link 1 of
length ``l1`` along ``+x`` at ``theta1 = 0``, joint 2 at the end of link 1 turning
about ``+z``, and link 2 of length ``l2``. Both angles are radians with the
right-hand rule about ``+z``; both link lengths are arguments, in metres, so no
fixed numeric arm is baked in.

The forward kinematics is written as the single-pass product of two link
transforms, ``A1(theta1) A2(theta2)``, and reproduces the trigonometric closed
form ``(l1 cos(t1) + l2 cos(t1 + t2), l1 sin(t1) + l2 sin(t1 + t2))``. The
Jacobian is the analytical positional Jacobian of that closed form, the planar
shadow of MR Eq. (5.1). The inverse kinematics is the standard 2R law with the
two elbow branches. This is a means module: it does not import ``mujoco`` and
does not know about the SO-101 model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

__all__ = [
    "PLANAR_BOUNDARY_GUARD",
    "planar_forward_kinematics",
    "planar_inverse_kinematics",
    "planar_ik_solutions",
    "planar_jacobian",
    "planar_manipulability",
    "planar_reachable",
]

_DTYPE = np.float64

#: Tolerance on ``|D|`` in the 2R law, where ``D`` is the cosine of the elbow
#: angle. It only absorbs floating-point rounding of a target that lies exactly
#: on the reachable boundary; it is **provisional** (pending Yohei approval) and
#: is not a physical workspace margin.
PLANAR_BOUNDARY_GUARD: Final[float] = 1e-12


def _as_length(value: object, name: str) -> float:
    """Return ``value`` as a finite, strictly positive link length."""
    try:
        length = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(length):
        raise ValueError(f"{name} must be finite")
    if length <= 0.0:
        raise ValueError(f"{name} must be positive, got {length!r}")
    return length


def _as_pair(value: object, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(2,)`` float64 array."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float64 array") from error
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_joint_angles(value: object) -> np.ndarray:
    """Return ``value`` as a finite ``(2,)`` float64 angle array."""
    return _as_pair(value, "joint_angles")


def _link_transform(theta: float, length: float) -> np.ndarray:
    """Return the planar link transform ``Trans_x(length) Rot_z(theta)``."""
    cos, sin = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [cos, -sin, length * cos],
            [sin, cos, length * sin],
            [0.0, 0.0, 1.0],
        ],
        dtype=_DTYPE,
    )


def planar_forward_kinematics(
    joint_angles: Sequence[float], l1: float, l2: float
) -> np.ndarray:
    """Return the ``(3, 3)`` planar homogeneous pose of the second link tip.

    The result is ``A1(theta1) A2(theta2)`` with ``A_i = Trans_x(l_i)
    Rot_z(theta_i)``, i.e. the pose of the tool frame in the base frame. Its
    translation is the trigonometric closed form
    ``(l1 cos(t1) + l2 cos(t1 + t2), l1 sin(t1) + l2 sin(t1 + t2))``.

    Raises
    ------
    ValueError
        If ``joint_angles`` is not a finite 2-vector, or ``l1``/``l2`` is not a
        finite positive length.
    """
    theta = _as_joint_angles(joint_angles)
    first = _as_length(l1, "l1")
    second = _as_length(l2, "l2")
    return _link_transform(float(theta[0]), first) @ _link_transform(float(theta[1]), second)


def planar_jacobian(
    joint_angles: Sequence[float], l1: float, l2: float
) -> np.ndarray:
    """Return the ``(2, 2)`` analytical positional Jacobian ``d(x, y) / d(t1, t2)``.

    With ``t12 = t1 + t2``::

        [[-l1 sin(t1) - l2 sin(t12), -l2 sin(t12)],
         [ l1 cos(t1) + l2 cos(t12),  l2 cos(t12)]]

    This is the planar positional part of the space Jacobian of MR Eq. (5.1):
    the first column is the velocity of the tip for unit ``t1`` and the second
    for unit ``t2``.
    """
    theta = _as_joint_angles(joint_angles)
    first = _as_length(l1, "l1")
    second = _as_length(l2, "l2")
    t1 = float(theta[0])
    t12 = float(theta[0] + theta[1])
    sin1, cos1 = math.sin(t1), math.cos(t1)
    sin12, cos12 = math.sin(t12), math.cos(t12)
    return np.array(
        [
            [-first * sin1 - second * sin12, -second * sin12],
            [first * cos1 + second * cos12, second * cos12],
        ],
        dtype=_DTYPE,
    )


def planar_reachable(point_xy: Sequence[float], l1: float, l2: float) -> bool:
    """Return whether the annulus ``|l1 - l2| <= r <= l1 + l2`` contains the point.

    The boundary is reachable and returns ``True``. Only
    :data:`PLANAR_BOUNDARY_GUARD` of slack is applied, and only so that a target
    computed to lie exactly on the boundary is not rejected by rounding.

    Raises
    ------
    ValueError
        If ``point_xy`` is not a finite 2-vector, or ``l1``/``l2`` is not a
        finite positive length.
    """
    point = _as_pair(point_xy, "point_xy")
    first = _as_length(l1, "l1")
    second = _as_length(l2, "l2")
    radius = float(np.hypot(point[0], point[1]))
    inner = abs(first - second)
    outer = first + second
    return bool(inner - PLANAR_BOUNDARY_GUARD <= radius <= outer + PLANAR_BOUNDARY_GUARD)


def _elbow_solutions(point: np.ndarray, l1: float, l2: float) -> tuple[float, float] | None:
    """Return the two elbow angles ``(theta2_up, theta2_down)`` or ``None``.

    ``theta2_up`` is non-negative and ``theta2_down`` its negative. ``None``
    means ``|D| > 1`` beyond :data:`PLANAR_BOUNDARY_GUARD`, i.e. unreachable.
    """
    radius_squared = float(point[0] ** 2 + point[1] ** 2)
    cosine = (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if abs(cosine) > 1.0 + PLANAR_BOUNDARY_GUARD:
        return None
    cosine = float(np.clip(cosine, -1.0, 1.0))
    theta2 = math.atan2(math.sqrt(max(0.0, 1.0 - cosine * cosine)), cosine)
    return theta2, -theta2


def planar_inverse_kinematics(
    point_xy: Sequence[float],
    l1: float,
    l2: float,
    *,
    elbow: str = "up",
) -> np.ndarray | None:
    """Return a joint-angle solution for a planar 2R tip at ``point_xy``.

    ``elbow="up"`` selects the non-negative ``theta2`` branch and
    ``elbow="down"`` the negative one. Unreachable targets return ``None``; they
    are not an exception (P9). The returned angles satisfy
    ``planar_forward_kinematics(q, l1, l2)`` to rounding.

    Raises
    ------
    ValueError
        If the inputs are malformed or ``elbow`` is not ``"up"`` or ``"down"``.
    """
    if elbow not in ("up", "down"):
        raise ValueError(f"elbow must be 'up' or 'down', got {elbow!r}")
    point = _as_pair(point_xy, "point_xy")
    first = _as_length(l1, "l1")
    second = _as_length(l2, "l2")

    solutions = _elbow_solutions(point, first, second)
    if solutions is None:
        return None
    theta2 = solutions[0] if elbow == "up" else solutions[1]

    theta1 = math.atan2(point[1], point[0]) - math.atan2(
        second * math.sin(theta2), first + second * math.cos(theta2)
    )
    return np.array([theta1, theta2], dtype=_DTYPE)


def planar_ik_solutions(
    point_xy: Sequence[float], l1: float, l2: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return both elbow solutions ``(up, down)`` or ``None`` when unreachable.

    On the reachable boundary the two branches coincide, so the two returned
    arrays are equal rather than distinct.

    Raises
    ------
    ValueError
        If ``point_xy`` is not a finite 2-vector, or ``l1``/``l2`` is not a
        finite positive length.
    """
    point = _as_pair(point_xy, "point_xy")
    first = _as_length(l1, "l1")
    second = _as_length(l2, "l2")

    solutions = _elbow_solutions(point, first, second)
    if solutions is None:
        return None
    theta1 = math.atan2(point[1], point[0])
    branches = []
    for theta2 in solutions:
        branches.append(
            np.array(
                [
                    theta1
                    - math.atan2(
                        second * math.sin(theta2), first + second * math.cos(theta2)
                    ),
                    theta2,
                ],
                dtype=_DTYPE,
            )
        )
    return branches[0], branches[1]


def planar_manipulability(
    joint_angles: Sequence[float], l1: float, l2: float
) -> float:
    """Return the Yoshikawa measure of the ``(2, 2)`` positional Jacobian.

    ``w = sqrt(det(J J^T)) = |det J| = |l1 l2 sin(theta2)|``, so it is exactly
    zero at ``theta2 = 0`` and ``theta2 = pi`` (the 2R singularities) and maximal
    at ``theta2 = pi / 2``.

    Raises
    ------
    ValueError
        If ``joint_angles`` is not a finite 2-vector, or ``l1``/``l2`` is not a
        finite positive length.
    """
    jacobian = planar_jacobian(joint_angles, l1, l2)
    return float(np.sqrt(max(0.0, float(np.linalg.det(jacobian @ jacobian.T)))))
