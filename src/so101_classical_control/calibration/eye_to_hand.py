"""Camera-to-base (eye-to-hand) calibration for the desk-fixed ``top_cam``.

This module recovers the pose of the desk-fixed camera in the robot base frame
from ground-truth marker poses. It is numpy only (``float64``), it does not import
``mujoco``, and it does not touch images: every pose is a ground-truth frame read
from the backend (``RobotBackend.get_camera_pose`` and
``RobotBackend.get_site_pose``) or supplied by the caller. The wrist-camera
eye-in-hand problem of Issue #8 is the other module in this package,
``hand_eye.py``.

Frames
------
``{base}``
    The robot base, the ``baseframe`` site. It is also the world frame of this
    scene, because the robot base is fixed to the desk.
``{C}``
    The desk-fixed ``top_cam``, in the **MuJoCo camera convention** (the camera
    looks along its own ``-Z``, ``+X`` right, ``+Y`` up), as returned by
    ``get_camera_pose``. See ``docs/cameras.md``.
``{M}``
    The ``ee_marker`` site, which is mounted on the fixed ``gripper`` body.
    ``T_base_marker = get_site_pose("ee_marker")``.
``{E}``
    The ``gripperframe`` site, the end effector used by the forward kinematics.
    It is a documented rigid intermediate only: ``ee_marker`` is fixed with
    respect to ``gripper``, so

    ``T_base_marker = T_base_gripperframe @ T_gripperframe_marker``

    with ``T_gripperframe_marker = inv(T_gripperframe) @ T_ee_marker`` constant
    across arm configurations. That rigidity is Issue #7's claim and is checked
    by the test. The solver does not have to go through ``{E}``; using ``{M}``
    directly is enough.

What is estimated
-----------------
``T_base_cam`` is the transform **from the camera frame to the base frame**: it
maps points expressed in ``{C}`` to ``{base}``. It is also the pose of the camera
in the base frame, i.e. exactly the object returned by
``get_camera_pose("top_cam")``, which is the ground truth this module is compared
against. Because ``top_cam`` is desk-fixed, ``T_base_cam`` is constant while the
arm moves (checked in Issue #7).

Closed-loop transform equation
------------------------------
Source: L02 lecture slides §2.7, the closed-loop transform equation, and its
Example 2.7.1 "Robot Reaching for a Bolt", together with the
"transform equation" entry of the L02 glossary, all in
``PhysicalAI-Applied2-202611/L02_20261119`` of
``yohei1126/utokyo-ai-courses``. Two paths from ``{base}`` to the same frame are
equal, and an unknown transform is isolated by multiplying by an inverse
(compound transformation and inverse transform). At every arm configuration::

    T_base_cam @ T_cam_marker = T_base_marker
    T_base_cam = T_base_marker @ inv(T_cam_marker)

``T_cam_marker`` is the pose of the marker in the camera frame — what image-based
marker detection would eventually provide (Issue #20, not implemented). In this
noiseless simulation it is formed algebraically from ground-truth poses as::

    T_cam_marker = inv(T_base_cam_gt) @ T_base_marker

so the closed-loop step is a consistency check of the frame algebra, not an
independent measurement, exactly as the eye-in-hand test of Issue #8 forms its
``A`` and ``B`` from ground-truth ``get_camera_pose`` / ``get_site_pose``
results. No marker is detected in an image anywhere in this module.

One configuration is enough for the closed-loop formula. With many
configurations every closed-loop estimate must agree with the ground truth and
with the others; this module does **not** average SE(3) poses, because no L02
source defines such an average.

Reusing the ``A X = X B`` solver
-------------------------------
The same loop gives an ``A X = X B`` problem that the existing
``hand_eye.solve_ax_xb`` can solve from many configurations, as a second
estimate. With the marker poses of configurations ``i`` and ``j``::

    A = T_base_marker_j @ inv(T_base_marker_i)
    B = T_cam_marker_j  @ inv(T_cam_marker_i)
    X = T_base_cam

so that ``A X = X B``. Note the product order ``T_j @ inv(T_i)``: this is the
motion from configuration ``i`` to configuration ``j`` expressed in the fixed
frame, not the eye-in-hand relative motion ``inv(T_i) @ T_j`` that
``hand_eye.build_motion_pairs`` and ``hand_eye.relative_motion`` build. Feeding
the eye-in-hand pairing to ``solve_ax_xb`` here does not recover ``T_base_cam``;
the test asserts that it fails. The two conventions therefore stay in separate
helpers, and ``solve_ax_xb`` itself is unchanged.
:func:`build_fixed_frame_motion_pairs` builds the ``T_j @ inv(T_i)`` lists.

Scope
-----
Course scope is L01-L02: rigid transforms and the closed-loop transform
equation. There is no inverse kinematics, no trajectory generation, no dynamics,
no controller, no camera intrinsic matrix, no marker detection, and no noise
model here. Noise robustness is Issue #10.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..kinematics import inverse_se3
from .hand_eye import _as_pose_list, _as_se3

__all__ = [
    "build_fixed_frame_motion_pairs",
    "camera_to_base",
    "camera_to_base_per_pose",
]


# ---------------------------------------------------------------------------
# Closed-loop transform equation
# ---------------------------------------------------------------------------


def camera_to_base(base_marker: np.ndarray, marker_in_camera: np.ndarray) -> np.ndarray:
    """Return ``T_base_cam = T_base_marker @ inv(T_cam_marker)``.

    Parameters
    ----------
    base_marker
        ``T_base_marker``, the ``(4, 4)`` pose of ``ee_marker`` in the base
        frame, e.g. from ``get_site_pose("ee_marker")``.
    marker_in_camera
        ``T_cam_marker``, the ``(4, 4)`` pose of the same marker in the camera
        frame. In this noiseless simulation it is formed algebraically as
        ``inv(T_base_cam_gt) @ T_base_marker``; no image is involved.

    Returns
    -------
    numpy.ndarray
        The ``(4, 4)`` float64 ``T_base_cam``, the camera pose in (and
        camera-to-base transform of) the base frame.

    Raises
    ------
    ValueError
        If either input is not a valid ``(4, 4)`` SE(3) transform.
    """
    base_marker_pose = _as_se3(base_marker, "base_marker")
    marker_in_camera_pose = _as_se3(marker_in_camera, "marker_in_camera")
    return base_marker_pose @ inverse_se3(marker_in_camera_pose)


def camera_to_base_per_pose(
    base_markers: Sequence[np.ndarray], marker_in_camera_poses: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Apply :func:`camera_to_base` to every configuration.

    ``base_markers[k]`` and ``marker_in_camera_poses[k]`` are the
    ``T_base_marker`` and ``T_cam_marker`` of the same arm configuration. The
    returns are one closed-loop estimate per configuration, in input order; none
    is an average over the others.

    Raises
    ------
    ValueError
        If the inputs are empty, not valid SE(3) transforms, or of different
        lengths.
    """
    base = _as_pose_list(base_markers, "base_markers")
    camera = _as_pose_list(marker_in_camera_poses, "marker_in_camera_poses")
    if len(base) != len(camera):
        raise ValueError(
            "base_markers and marker_in_camera_poses must have the same length, "
            f"got {len(base)} and {len(camera)}"
        )
    return [
        base_pose @ inverse_se3(camera_pose)
        for base_pose, camera_pose in zip(base, camera, strict=True)
    ]


# ---------------------------------------------------------------------------
# Fixed-frame motion pairs for A X = X B
# ---------------------------------------------------------------------------


def build_fixed_frame_motion_pairs(
    base_markers: Sequence[np.ndarray], marker_in_camera_poses: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build ``(A_list, B_list)`` with ``A = T_j @ inv(T_i)`` from pose sequences.

    ``base_markers[i]`` is ``T_base_marker`` and ``marker_in_camera_poses[i]`` is
    ``T_cam_marker`` of the same arm configuration. All unordered pairs
    ``i < j`` are used, in lexicographic order, with::

        A = base_markers[j] @ inv(base_markers[i])
        B = marker_in_camera_poses[j] @ inv(marker_in_camera_poses[i])

    so that ``solve_ax_xb(A_list, B_list)`` returns ``X = T_base_cam``.

    This is deliberately not ``hand_eye.build_motion_pairs``: that helper builds
    the eye-in-hand convention ``inv(T_i) @ T_j``, which does not recover
    ``T_base_cam`` here.

    Returns two lists of the same length; pair ``k`` is ``(A_list[k], B_list[k])``.

    Raises
    ------
    ValueError
        If the inputs are empty, not valid SE(3) transforms, of different
        lengths, or shorter than two configurations.
    """
    base = _as_pose_list(base_markers, "base_markers")
    camera = _as_pose_list(marker_in_camera_poses, "marker_in_camera_poses")
    if len(base) != len(camera):
        raise ValueError(
            "base_markers and marker_in_camera_poses must have the same length, "
            f"got {len(base)} and {len(camera)}"
        )
    if len(base) < 2:
        raise ValueError("at least two pose pairs are needed to form a relative motion")

    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            a_list.append(base[j] @ inverse_se3(base[i]))
            b_list.append(camera[j] @ inverse_se3(camera[i]))
    return a_list, b_list
