"""Standard (classical) Denavit-Hartenberg forward kinematics for the SO-101 arm.

The model uses the standard/classical DH convention of L02 exercise-notebook
sections 8-9 (not modified DH)::

    A_i = Rot_z(theta_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i)

with parameters in the order ``a`` (link length along ``x_i``, metres),
``alpha`` (link twist about ``x_i``, radians), ``d`` (offset along ``z_{i-1}``,
metres), and ``theta`` (joint angle about ``z_{i-1}``, radians). For a revolute
joint ``theta_i = q_i + theta_offset_i`` with constant ``d_i``.

The fixed base and tool transforms stay separate from the five joint rows, as
in L02 sections 8.1 and 8.3::

    T_base_ee(q) = T_BASE_TO_0 @ A_1(q_1) ... A_5(q_5) @ T_N_TO_EE

The end effector is the MuJoCo site ``gripperframe``. The gripper DoF is
excluded from the arm chain: this module models the five arm joints
:data:`ARM_JOINT_NAMES` only.

Source of the numbers
---------------------
Every constant below was derived from the vendored model
``src/so101_classical_control/sim/assets/robotstudio_so101/so101.xml``
(MuJoCo Menagerie commit ``370c43965323c0944b433b202aac5058b5f907a9``,
Apache License 2.0, recorded in ``sim/assets/PROVENANCE.md``). No published
third-party SO-101 tables were used. The relevant elements are:

- bodies ``base``, ``shoulder``, ``upper_arm``, ``lower_arm``, ``wrist``,
  ``gripper`` with their compiled ``pos``/``quat``;
- hinge joints ``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``,
  ``wrist_flex`` (all local axis ``(0, 0, 1)``), ``wrist_roll``, and the
  excluded ``gripper`` joint;
- site ``gripperframe`` on the ``gripper`` body.

The frame assignment was read off the compiled model at ``q = 0``: the
``z_{i-1}`` axis is joint ``i``'s world axis, ``x_i`` is the common normal
between consecutive joint axes (the link direction for the three parallel
``(0, 1, 0)`` axes), and the fifth DH frame coincides with the fourth. Then
``T_BASE_TO_0`` is the base frame at the pan joint and ``T_N_TO_EE`` is the
fixed transform from the fifth DH frame to ``gripperframe``.

Geometry that the frozen numbers keep and that must not be "cleaned up":

- the ``upper_arm`` Y offset of ``-0.0182778`` m (18.3 mm), which moves the
  ``wrist_roll`` axis in Y and therefore appears as the small
  ``d_4 = -0.0001778`` m; removing the offset would change ``d_4`` to
  ``0.0181`` m and shift the q = 0 tool pose by 18.3 mm in Y;
- the pan axis pointing along world ``-Z`` at ``q = 0`` (``shoulder`` has a
  180 degree rotation about Y), which ``T_BASE_TO_0`` reproduces as
  ``diag(1, -1, -1)``;
- the gripper mount quaternion, which is not a clean 90 degree rotation and
  leaves the tool frame tilted by about 2.8 degrees about X. The tilt lives in
  ``T_N_TO_EE``; the q = 0 tool rotation is not the identity.

At ``q = 0`` the resulting pose is the ``gripperframe`` pose reported by
MuJoCo: translation ``(0.3913615, -0.0009794652485835523,
0.24634517669302677)`` m and a rotation of ``0.048679514853476276`` rad
(2.789 degrees) about ``+X``.

Only ``numpy`` is imported here. This module does not import ``mujoco``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from ..backend import JOINT_NAMES
from .transforms import compose_se3_body, pose, rx, rz

__all__ = [
    "ARM_JOINT_NAMES",
    "arm_joint_angles",
    "SO101_DH_PARAMS",
    "SO101_T_BASE_TO_0",
    "SO101_T_N_TO_EE",
    "dh_transform",
    "forward_kinematics",
]

_DTYPE = np.float64

#: The five arm joint names in :data:`~so101_classical_control.backend.JOINT_NAMES`
#: order. ``gripper`` is deliberately excluded: it does not move the
#: ``gripperframe`` site and is not part of the arm chain.
ARM_JOINT_NAMES: Final[tuple[str, ...]] = JOINT_NAMES[:5]

#: Frozen standard-DH rows ``(a, alpha, d, theta_offset)`` for the five arm
#: joints, in metres and radians. Derived from the vendored ``so101.xml``.
SO101_DH_PARAMS: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.0303992, math.pi / 2.0, -0.0542, 0.0),
    (0.11600002112068772, 0.0, 0.0, -1.3270094336148002),
    (0.13500018518505819, 0.0, 0.0, 1.2884814367726616),
    (0.0, -math.pi / 2.0, -0.0001778, 1.6093243236370354),
    (0.0, 0.0, 0.0, 0.0),
)

#: ``T_b^0``: pose of DH frame 0 relative to the robot base (``baseframe``).
#: The rotation is ``diag(1, -1, -1)`` because the shoulder body is rotated by
#: 180 degrees about Y, so positive pan turns about world ``-Z`` at ``q = 0``.
SO101_T_BASE_TO_0: Final[np.ndarray] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0388353],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0624],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=_DTYPE,
)

#: ``T_n^e``: fixed transform from the fifth DH frame (the ``wrist_roll`` axis)
#: to the ``gripperframe`` site. The rotation is not a clean 90 degrees: the
#: gripper mount leaves a tilt of about 2.8 degrees about X.
SO101_T_N_TO_EE: Final[np.ndarray] = np.array(
    [
        [0.0, -0.04866029119613833, -0.9988153863756335, -0.011975176693026862],
        [0.0, -0.9988153863756337, 0.04866029119613834, 0.0008016652485835482],
        [-1.0, 0.0, 0.0, -0.159227],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=_DTYPE,
)


def _as_angle(value: object, name: str) -> float:
    """Return ``value`` as a finite Python float or raise ``ValueError``."""
    try:
        angle = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not np.isfinite(angle):
        raise ValueError(f"{name} must be finite")
    return angle


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Return the standard DH transform ``A_i`` as a ``(4, 4)`` float64 array.

    ``A_i = Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)``, which expands to
    the L02 matrix::

        [[ct, -st*ca,  st*sa, a*ct],
         [st,  ct*ca, -ct*sa, a*st],
         [ 0,      sa,     ca,    d],
         [ 0,       0,      0,    1]]

    ``a`` and ``d`` are in metres, ``alpha`` and ``theta`` in radians. It is
    built from the :mod:`~so101_classical_control.kinematics.transforms`
    helpers rather than a hand-expanded matrix.
    """
    a = _as_angle(a, "a")
    alpha = _as_angle(alpha, "alpha")
    d = _as_angle(d, "d")
    theta = _as_angle(theta, "theta")
    along_z = pose(rz(theta), (0.0, 0.0, d))
    along_x = pose(rx(alpha), (a, 0.0, 0.0))
    return compose_se3_body(along_z, along_x)


def arm_joint_angles(q: Sequence[float] | Mapping[str, float]) -> tuple[float, ...]:
    """Return the five arm angles in :data:`ARM_JOINT_NAMES` order.

    A mapping must contain every arm joint; a ``gripper`` entry is accepted and
    ignored. A sequence may have length five, or length six in
    :data:`~so101_classical_control.backend.JOINT_NAMES` order, in which case
    the final ``gripper`` entry is ignored.
    """
    if isinstance(q, Mapping):
        unknown = sorted(set(q) - set(ARM_JOINT_NAMES) - {"gripper"})
        if unknown:
            raise ValueError(f"unknown joint names: {', '.join(map(str, unknown))}")
        missing = [name for name in ARM_JOINT_NAMES if name not in q]
        if missing:
            raise ValueError(f"missing arm joints: {', '.join(missing)}")
        values: Sequence[object] = [q[name] for name in ARM_JOINT_NAMES]
    else:
        try:
            array = np.asarray(q, dtype=_DTYPE)
        except (TypeError, ValueError) as error:
            raise ValueError("q must be a sequence of radians or a joint mapping") from error
        if array.ndim != 1:
            raise ValueError(f"q must be one-dimensional, got shape {array.shape}")
        if array.shape[0] == len(JOINT_NAMES):
            array = array[: len(ARM_JOINT_NAMES)]
        elif array.shape[0] != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"q must have {len(ARM_JOINT_NAMES)} entries "
                f"({len(JOINT_NAMES)} with a trailing gripper entry), got {array.shape[0]}"
            )
        values = list(array)

    return tuple(
        _as_angle(value, f"q[{name}]")
        for name, value in zip(ARM_JOINT_NAMES, values, strict=True)
    )


def forward_kinematics(q: Sequence[float] | Mapping[str, float]) -> np.ndarray:
    """Return the ``gripperframe`` pose in the robot base frame.

    ``q`` is either a sequence of five arm angles in :data:`ARM_JOINT_NAMES`
    order (radians), a sequence of six in
    :data:`~so101_classical_control.backend.JOINT_NAMES` order, or a mapping
    keyed by joint names. In the last two forms the ``gripper`` value is
    ignored because the gripper DoF does not move ``gripperframe``. Unknown
    joint names and missing arm joints raise ``ValueError``; so do non-finite
    or wrongly shaped inputs.

    The returned ``(4, 4)`` float64 homogeneous transform is
    ``T_BASE_TO_0 @ A_1 ... A_5 @ T_N_TO_EE``.
    """
    angles = arm_joint_angles(q)
    transform = SO101_T_BASE_TO_0.copy()
    for (a, alpha, d, theta_offset), angle in zip(
        SO101_DH_PARAMS, angles, strict=True
    ):
        transform = compose_se3_body(
            transform, dh_transform(a, alpha, d, theta_offset + angle)
        )
    return compose_se3_body(transform, SO101_T_N_TO_EE)
