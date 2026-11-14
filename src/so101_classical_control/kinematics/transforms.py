"""SO(3) and SE(3) transform utilities for the SO-101 arm.

This module provides small, explicit helpers built on ``numpy`` only. All
arrays are ``float64`` and all angles are radians with the right-hand rule.

Conventions
-----------
Quaternions
    Unit quaternions are arrays of shape ``(4,)`` in the order ``(w, x, y, z)``
    (scalar first), the same order MuJoCo uses and the same ``Q = {eta, eps}``
    form as L02. Quaternions returned by this module are canonicalized: the
    sign is flipped when necessary so that ``w >= 0``. If ``w == 0`` the sign
    is chosen so that the first nonzero component of ``(x, y, z)`` is positive.
    Incoming quaternions may have either sign; they are normalized before use.

Roll-pitch-yaw (RPY)
    Z-Y-X intrinsic/extrinsic sequence ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``,
    i.e. rotation about X by ``roll``, about Y by ``pitch``, and about Z by
    ``yaw``. Angles are radians.

Composition
    Composition helpers state in their names which frame the increment lives
    in. The ``*_fixed`` helpers pre-multiply (``R_fixed @ R``), which applies
    an increment expressed in the fixed/world frame. The ``*_body`` helpers
    post-multiply (``R @ R_body``), which applies an increment expressed in the
    body/current frame. Both helpers are plain matrix products, so the result
    equals left argument @ right argument in each case; the names describe the
    frame, not a different product formula.

L02 naming note
    The L02 exercise notebook labels the Z-axis angle of the same Z-Y-X
    sequence ``roll`` (phi) and the X-axis angle ``yaw`` (psi). That labeling
    is the opposite of this module and of standard aerospace RPY. The matrix
    sequence is identical; callers translating L02 examples must swap the two
    *names* and must not change the Z-Y-X multiplication order.

Singularities
    ``rotation_to_rpy`` pins ``roll = 0`` at gimbal lock (``pitch = +/- pi/2``)
    and absorbs the remaining freedom into ``yaw``. ``rotation_to_zyz`` pins
    ``psi = 0`` at its singularities (``theta = 0`` or ``pi``) and absorbs the
    remaining freedom into ``phi``. ``rotation_to_axis_angle`` handles
    ``angle = pi`` by recovering the axis from the rotation matrix instead of
    raising (unlike L02's exercise helper).
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float64
# Tolerance used when validating that a matrix is a proper rotation.
_ORTHONORMAL_TOL = 1e-6
# Below this value the rotation angle is treated as 0 or pi.
_ANGLE_EPS = 1e-9

__all__ = [
    "axis_angle_to_rotation",
    "compose_quaternion_body",
    "compose_quaternion_fixed",
    "compose_se3_body",
    "compose_se3_fixed",
    "compose_so3_body",
    "compose_so3_fixed",
    "identity_se3",
    "identity_so3",
    "inverse_quaternion",
    "inverse_se3",
    "inverse_so3",
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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_scalar(value: float, name: str) -> float:
    """Return ``value`` as a finite Python float or raise ``ValueError``."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_vector(value: object, size: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(size,)`` float64 array."""
    array = np.asarray(value, dtype=_DTYPE)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_square_matrix(value: object, size: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(size, size)`` float64 array."""
    matrix = np.asarray(value, dtype=_DTYPE)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _as_rotation(value: object, name: str = "R") -> np.ndarray:
    """Return ``value`` as a proper rotation matrix or raise ``ValueError``."""
    rotation = _as_square_matrix(value, 3, name)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ORTHONORMAL_TOL):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ORTHONORMAL_TOL):
        raise ValueError(f"{name} must have determinant +1")
    return rotation


def _as_se3(value: object, name: str = "T") -> np.ndarray:
    """Return ``value`` as a homogeneous transform with a proper rotation."""
    transform = _as_square_matrix(value, 4, name)
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=_ORTHONORMAL_TOL):
        raise ValueError(f"{name} must have last row [0, 0, 0, 1]")
    _as_rotation(transform[:3, :3], f"{name}[:3, :3]")
    return transform


def _canonical_sign(quaternion: np.ndarray) -> np.ndarray:
    """Return ``quaternion`` with the documented canonical sign."""
    if quaternion[0] < 0.0:
        return -quaternion
    if quaternion[0] == 0.0:
        for component in quaternion[1:]:
            if component != 0.0:
                if component < 0.0:
                    return -quaternion
                break
    return quaternion


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Return ``axis`` with the first nonzero component non-negative."""
    for component in axis:
        if component != 0.0:
            return -axis if component < 0.0 else axis
    return axis


def _unit_quaternion(value: object, name: str = "q") -> np.ndarray:
    """Return ``value`` normalized to unit length with canonical sign."""
    quaternion = _as_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError(f"{name} must have nonzero norm")
    return _canonical_sign(quaternion / norm)


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product ``left * right`` for ``(w, x, y, z)`` quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=_DTYPE,
    )


# ---------------------------------------------------------------------------
# SO(3): elementary rotations, identity, inverse, composition
# ---------------------------------------------------------------------------


def rx(angle: float) -> np.ndarray:
    """Rotation about the X axis by ``angle`` radians (right-hand rule)."""
    angle = _as_scalar(angle, "angle")
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]],
        dtype=_DTYPE,
    )


def ry(angle: float) -> np.ndarray:
    """Rotation about the Y axis by ``angle`` radians (right-hand rule)."""
    angle = _as_scalar(angle, "angle")
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]],
        dtype=_DTYPE,
    )


def rz(angle: float) -> np.ndarray:
    """Rotation about the Z axis by ``angle`` radians (right-hand rule)."""
    angle = _as_scalar(angle, "angle")
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        dtype=_DTYPE,
    )


def identity_so3() -> np.ndarray:
    """Return the ``(3, 3)`` identity rotation."""
    return np.eye(3, dtype=_DTYPE)


def inverse_so3(rotation: np.ndarray) -> np.ndarray:
    """Return the inverse of a rotation matrix as its transpose."""
    return _as_rotation(rotation).T.copy()


def compose_so3_fixed(r_fixed: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Compose with an increment expressed in the fixed/world frame.

    Pre-multiplies: returns ``r_fixed @ rotation``.
    """
    return _as_rotation(r_fixed, "r_fixed") @ _as_rotation(rotation, "rotation")


def compose_so3_body(rotation: np.ndarray, r_body: np.ndarray) -> np.ndarray:
    """Compose with an increment expressed in the body/current frame.

    Post-multiplies: returns ``rotation @ r_body``.
    """
    return _as_rotation(rotation, "rotation") @ _as_rotation(r_body, "r_body")


# ---------------------------------------------------------------------------
# Quaternions, order (w, x, y, z)
# ---------------------------------------------------------------------------


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return ``quaternion`` scaled to unit length with canonical sign."""
    return _unit_quaternion(quaternion)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a rotation matrix.

    The input is normalized first, so non-unit quaternions and either sign
    convention are accepted.
    """
    w, x, y, z = _unit_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=_DTYPE,
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a canonical ``(w, x, y, z)`` quaternion.

    Uses the numerically stable branch on the largest diagonal entry rather
    than dividing by ``sqrt(trace + 1)`` alone, so rotations near ``pi`` are
    handled without loss of precision. The returned quaternion has ``w >= 0``;
    if ``w == 0`` the first nonzero of ``(x, y, z)`` is positive.
    """
    r = _as_rotation(rotation)
    trace = r[0, 0] + r[1, 1] + r[2, 2]

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        )
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        quaternion = np.array(
            [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        )
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        quaternion = np.array(
            [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        quaternion = np.array(
            [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ]
        )

    return _canonical_sign(quaternion / np.linalg.norm(quaternion))


def inverse_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return the inverse of a unit quaternion, canonicalized on return."""
    w, x, y, z = _unit_quaternion(quaternion)
    return _canonical_sign(np.array([w, -x, -y, -z], dtype=_DTYPE))


def compose_quaternion_fixed(q_fixed: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Compose with a fixed/world-frame increment using the Hamilton product.

    Returns the canonicalized product ``q_fixed * quaternion``; its rotation
    matrix equals ``quaternion_to_rotation(q_fixed) @ quaternion_to_rotation(quaternion)``.
    """
    left = _unit_quaternion(q_fixed, "q_fixed")
    right = _unit_quaternion(quaternion, "quaternion")
    return _unit_quaternion(_quaternion_product(left, right))


def compose_quaternion_body(quaternion: np.ndarray, q_body: np.ndarray) -> np.ndarray:
    """Compose with a body/current-frame increment using the Hamilton product.

    Returns the canonicalized product ``quaternion * q_body``; its rotation
    matrix equals ``quaternion_to_rotation(quaternion) @ quaternion_to_rotation(q_body)``.
    """
    left = _unit_quaternion(quaternion, "quaternion")
    right = _unit_quaternion(q_body, "q_body")
    return _unit_quaternion(_quaternion_product(left, right))


# ---------------------------------------------------------------------------
# Axis-angle
# ---------------------------------------------------------------------------


def axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert an axis-angle pair to a rotation matrix via Rodrigues' formula.

    ``axis`` need not be unit length but must be nonzero and finite; it is
    normalized internally. ``angle`` is in radians.
    """
    axis = _as_vector(axis, 3, "axis")
    angle = _as_scalar(angle, "angle")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must have nonzero norm")
    unit_axis = axis / norm
    k = np.array(
        [
            [0.0, -unit_axis[2], unit_axis[1]],
            [unit_axis[2], 0.0, -unit_axis[0]],
            [-unit_axis[1], unit_axis[0], 0.0],
        ],
        dtype=_DTYPE,
    )
    return np.eye(3, dtype=_DTYPE) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rotation_to_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a rotation matrix to a unit axis and an angle in ``[0, pi]``.

    For ``0 < angle < pi`` the axis is unique and no sign convention is
    applied. At ``angle = 0`` the conventional axis ``[1, 0, 0]`` is returned.
    At ``angle = pi`` the axis sign is ambiguous; the returned axis has its
    first nonzero component positive (a conventional sign). Unlike L02's
    exercise helper, ``angle = pi`` is handled instead of raising.
    """
    r = _as_rotation(rotation)
    skew = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=_DTYPE
    )
    sin_angle = 0.5 * float(np.linalg.norm(skew))
    cos_angle = float(np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0))

    if sin_angle < _ANGLE_EPS:
        if cos_angle > 0.0:
            return np.array([1.0, 0.0, 0.0], dtype=_DTYPE), 0.0
        axis = _axis_from_pi_rotation(r)
        return _canonical_axis(axis), float(np.pi)

    angle = float(np.arctan2(sin_angle, cos_angle))
    axis = skew / (2.0 * sin_angle)
    return axis / np.linalg.norm(axis), angle


def _axis_from_pi_rotation(rotation: np.ndarray) -> np.ndarray:
    """Recover the axis of a rotation by ``pi`` from its symmetric part."""
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    axis = np.zeros(3, dtype=_DTYPE)
    axis[index] = np.sqrt(max(0.0, (rotation[index, index] + 1.0) / 2.0))
    if axis[index] < _ANGLE_EPS:
        # Degenerate only for a non-rotation; validation already excludes it.
        axis[index] = 1.0
        return axis
    for other in range(3):
        if other != index:
            axis[other] = rotation[index, other] / (2.0 * axis[index])
    return axis / np.linalg.norm(axis)


# ---------------------------------------------------------------------------
# RPY (Z-Y-X)
# ---------------------------------------------------------------------------


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compose roll, pitch, and yaw into ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    ``roll`` is the rotation about X, ``pitch`` about Y, and ``yaw`` about Z.

    L02 naming note: the L02 exercise notebook labels the Z-axis angle of this
    same Z-Y-X sequence ``roll`` (phi) and the X-axis angle ``yaw`` (psi). That
    labeling is the opposite of this module and of standard aerospace RPY.
    Callers translating L02 examples must swap the two names and must not
    change the Z-Y-X multiplication order.
    """
    return rz(yaw) @ ry(pitch) @ rx(roll)


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` with ``pitch`` principal.

    ``pitch`` lies in ``[-pi/2, pi/2]``. At gimbal lock (``pitch = +/- pi/2``)
    roll and yaw are not unique; this function sets ``roll = 0`` and absorbs
    the remaining freedom into ``yaw``, so
    ``rpy_to_rotation(*rotation_to_rpy(R))`` still recovers ``R``.

    L02 naming note: the L02 exercise notebook labels the Z-axis angle of this
    same Z-Y-X sequence ``roll`` (phi) and the X-axis angle ``yaw`` (psi). That
    labeling is the opposite of this module and of standard aerospace RPY.
    Callers translating L02 examples must swap the two names and must not
    change the Z-Y-X multiplication order.
    """
    r = _as_rotation(rotation)
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))
    # |cos(pitch)| is a well-conditioned distance from gimbal lock.
    cos_pitch = float(np.hypot(r[0, 0], r[1, 0]))

    if cos_pitch > _ANGLE_EPS:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        return roll, pitch, yaw

    # Gimbal lock: roll is not observable, so pin roll = 0.
    sign = 1.0 if sin_pitch > 0.0 else -1.0
    yaw = float(np.arctan2(sign * r[1, 2], r[1, 1]))
    return 0.0, pitch, yaw


# ---------------------------------------------------------------------------
# ZYZ Euler
# ---------------------------------------------------------------------------


def zyz_to_rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    """Compose ZYZ Euler angles into ``R = Rz(phi) @ Ry(theta) @ Rz(psi)``."""
    return rz(phi) @ ry(theta) @ rz(psi)


def rotation_to_zyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose ``R = Rz(phi) @ Ry(theta) @ Rz(psi)`` with ``theta`` in ``[0, pi]``.

    ZYZ is singular at ``theta = 0`` and ``theta = pi``, where only a
    combination of ``phi`` and ``psi`` is observable. In both cases this
    function pins ``psi = 0`` and puts the remaining freedom into ``phi``, so
    ``zyz_to_rotation(*rotation_to_zyz(R))`` still recovers ``R``.
    """
    r = _as_rotation(rotation)
    sin_theta = float(np.hypot(r[0, 2], r[1, 2]))
    theta = float(np.arctan2(sin_theta, r[2, 2]))

    if sin_theta < _ANGLE_EPS:
        phi = float(np.arctan2(r[1, 0], r[0, 0]))
        return phi, theta, 0.0

    phi = float(np.arctan2(r[1, 2], r[0, 2]))
    psi = float(np.arctan2(r[2, 1], -r[2, 0]))
    return phi, theta, psi


# ---------------------------------------------------------------------------
# SE(3): homogeneous transforms
# ---------------------------------------------------------------------------


def pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a rotation and a translation."""
    r = _as_rotation(rotation)
    p = _as_vector(translation, 3, "translation")
    transform = np.eye(4, dtype=_DTYPE)
    transform[:3, :3] = r
    transform[:3, 3] = p
    return transform


def identity_se3() -> np.ndarray:
    """Return the ``(4, 4)`` identity homogeneous transform."""
    return np.eye(4, dtype=_DTYPE)


def rotation_of(transform: np.ndarray) -> np.ndarray:
    """Return the ``(3, 3)`` rotation block of a homogeneous transform."""
    return _as_se3(transform)[:3, :3].copy()


def translation_of(transform: np.ndarray) -> np.ndarray:
    """Return the ``(3,)`` translation block of a homogeneous transform."""
    return _as_se3(transform)[:3, 3].copy()


def inverse_se3(transform: np.ndarray) -> np.ndarray:
    """Return the inverse as ``[R.T, -R.T @ p]`` without a numeric inversion."""
    t = _as_se3(transform)
    r = t[:3, :3]
    p = t[:3, 3]
    inverse = np.eye(4, dtype=_DTYPE)
    inverse[:3, :3] = r.T
    inverse[:3, 3] = -r.T @ p
    return inverse


def compose_se3_fixed(t_fixed: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Compose with an increment expressed in the fixed/world frame.

    Pre-multiplies: returns ``t_fixed @ transform``.
    """
    return _as_se3(t_fixed, "t_fixed") @ _as_se3(transform, "transform")


def compose_se3_body(transform: np.ndarray, t_body: np.ndarray) -> np.ndarray:
    """Compose with an increment expressed in the body/current frame.

    Post-multiplies: returns ``transform @ t_body``.
    """
    return _as_se3(transform, "transform") @ _as_se3(t_body, "t_body")


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Map a 3-vector as a point: ``R @ point + t``."""
    t = _as_se3(transform)
    p = _as_vector(point, 3, "point")
    return t[:3, :3] @ p + t[:3, 3]
