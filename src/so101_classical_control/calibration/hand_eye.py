"""Eye-in-hand hand-eye calibration ``A X = X B`` for the SO-101 wrist camera.

This module recovers the fixed rigid transform between the arm end effector and
an arm-mounted camera from relative motions only. It is numpy only (``float64``),
it does not import ``mujoco``, and it does not touch images: every pose is a
ground-truth frame read from the backend (``RobotBackend.get_site_pose`` and
``RobotBackend.get_camera_pose``) or supplied by the caller.

Conventions
-----------
``X`` is the transform **from the wrist-camera frame to the end-effector frame**,
written ``T_gripperframe->wrist_cam`` in this repository::

    X = inv(T_gripperframe) @ T_wrist_cam

so that ``T_wrist_cam = T_gripperframe @ X`` for every arm configuration. This is
the ``X`` of L02 §10 ("Hand-Eye Calibration"): the fixed transform between the
tool and the camera mounted on it.

For two arm configurations ``i`` and ``j`` the relative motions are::

    A = inv(T_gripperframe_i) @ T_gripperframe_j
    B = inv(T_camera_i)       @ T_camera_j

which satisfy ``A X = X B`` exactly for a rigid ``X``. ``B`` is built from camera
poses, not from the static ``desk_marker`` site. The marker form
``B = T_cam_marker_i @ inv(T_cam_marker_j)``, with
``T_cam_marker = inv(T_cam) @ T_desk_marker``, is the same matrix because the
marker is static (note the product order: it is not
``inv(T_cam_marker_i) @ T_cam_marker_j``), but camera poses do not depend on the
marker being inside the camera frustum.

All poses use the **MuJoCo camera convention** — the camera looks along its own
``-Z``, with ``+X`` to the right of the image and ``+Y`` up — because that is the
convention of ``get_camera_pose``. No conversion to the OpenCV/optical convention
is applied, for the solver or for the ground-truth comparison; a rotation of the
camera frame by ``diag(1, -1, -1)`` conjugates ``X`` consistently and would not
change the problem, but doing it in only one place would. See
``docs/cameras.md``.

Sources
-------
- L02 lecture notes, section "3. Hand-Eye Calibration (``A X = X B``)":
  ``PhysicalAI-Applied2-202611/L02_20261119/README.md`` in
  ``yohei1126/utokyo-ai-courses``. Eye-in-hand setup, the rotation equation
  ``R_A R_X = R_X R_B`` solved with SVD, and the translation equation
  ``(R_A - I) t_X = R_X t_B - t_A`` solved by linear least squares.
- L02 exercise notebook (solutions), section 10, cells 88-98, in the same
  directory: synthetic data generation (§10.1), the rotation SVD (§10.2), and the
  stacked least-squares translation (§10.3).
- Issue #8 in this repository: collect pose pairs consistent with eye-in-hand
  ``A X = X B``, compare the recovered ``X`` with the MuJoCo ground-truth
  wrist-camera transform, and keep the error on noiseless simulated data below a
  stated tight tolerance.
- ``docs/cameras.md`` and ``docs/calibration.md`` for the frames, the marker
  sites, and the axis convention of this repository.

Rotation solver
---------------
``R_A R_X = R_X R_B`` is equivalent to ``R_A = R_X R_B R_X^T``, so the two
relative rotations have the same angle and the axis of ``R_A`` is ``R_X`` times
the axis of ``R_B``. In rotation-vector form::

    log(R_A) = R_X log(R_B)

Each pair therefore contributes one vector correspondence. Writing
``omega_A = angle * axis`` (with ``rotation_to_axis_angle``) and stacking all of
them, the rotation is the orthogonal Procrustes problem of the matrix::

    H = sum_i omega_A_i omega_B_i^T
    H = U S V^T
    R_X = U V^T

with the third column of ``U`` negated when ``det(U V^T) < 0``, so that
``det(R_X) = +1``. At ``angle = 0`` the rotation vector is the zero vector, so a
zero-angle pair contributes nothing rather than the conventional ``[1, 0, 0]``
axis that ``rotation_to_axis_angle`` returns alongside ``angle = 0``.

Pairs are filtered before the rotation stage:

- relative rotations below :data:`MIN_RELATIVE_ROTATION_ANGLE` (0.01 rad) are
  dropped, because their rotation vector is too short to carry a well-conditioned
  axis;
- relative rotations above :data:`MAX_RELATIVE_ROTATION_ANGLE` (``pi - 0.01``) are
  dropped, because at exactly ``pi`` the sign of the axis is ambiguous and the
  rotation vector is discontinuous;
- fewer than :data:`MIN_USABLE_PAIRS` (3) surviving pairs raise ``ValueError``,
  as do usable pairs whose axes are all parallel (rank-deficient ``H``).

This axis-then-SVD construction is often called the Park-Martin method in the
literature. L02 does not use that name: the source for *this* module is the L02
equation plus SVD, as above, not a paper.

Note on notebook cell 92
------------------------
The notebook's own rotation estimate sums ``M = sum_i R_A_i R_B_i^T`` and takes
``R_X = U V^T`` from ``M = U S V^T``. That Procrustes problem minimizes
``sum_i ||R_A_i - R_X R_B_i||``, which is *not* the hand-eye equation; it only
coincides with it when ``R_X`` commutes with every ``R_B_i``. On the notebook's
own synthetic data it leaves an error of order ``1e-3`` rad, and on a
wrist-camera-sized ``X`` (rotation near 90 deg) it is wrong by order ``1`` rad.
This module therefore does **not** ship that expression; it uses the
rotation-vector correspondence above, which is exact for consistent data. The L02
README's claim that noiseless synthetic data is recovered to ``1e-15`` is not
reproduced by the notebook code; the synthetic test in
``tests/test_handeye_axxb.py`` checks this module against a stated tolerance
instead.

Translation solver
------------------
With ``R_X`` known, the translation part of ``A X = X B`` is linear in ``t_X``::

    (R_A - I) t_X = R_X t_B - t_A

The blocks are stacked for every pair — including the pairs the rotation stage
skipped, whose rows are merely uninformative — and solved with
``np.linalg.lstsq``, exactly as in L02 §10.3. ``X = pose(R_X, t_X)`` is returned
through the existing helper in ``kinematics.transforms``.

Scope
-----
Course scope is L01-L02: rigid transforms and hand-eye ``A X = X B``. There is no
inverse kinematics, no trajectory generation, no dynamics, no controller, no
camera intrinsic matrix, no marker detection, and no noise model here. Eye-to-hand
calibration of the desk camera is a separate issue, and the noise sweep is
another.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..kinematics import (
    inverse_se3,
    pose,
    rotation_of,
    rotation_to_axis_angle,
)

__all__ = [
    "MAX_RELATIVE_ROTATION_ANGLE",
    "MIN_RELATIVE_ROTATION_ANGLE",
    "MIN_USABLE_PAIRS",
    "build_motion_pairs",
    "relative_motion",
    "solve_ax_xb",
]

_DTYPE = np.float64

#: Relative rotations smaller than this are dropped from the rotation stage.
MIN_RELATIVE_ROTATION_ANGLE = 1e-2  # radians
#: Relative rotations larger than this are dropped from the rotation stage,
#: because the axis sign is ambiguous at ``pi``.
MAX_RELATIVE_ROTATION_ANGLE = float(np.pi) - 1e-2  # radians
#: Fewer usable pairs than this cannot determine the rotation.
MIN_USABLE_PAIRS = 3
#: A second singular value of ``H`` below this fraction of the first one means the
#: relative rotation axes are effectively collinear, so ``R_X`` is not determined.
_RANK_TOL = 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_se3(transform: object, name: str) -> np.ndarray:
    """Return ``transform`` as a validated ``(4, 4)`` float64 SE(3) array."""
    try:
        array = np.asarray(transform, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float64 array") from error
    try:
        # ``rotation_of`` validates shape, finiteness, the last row, orthonormality,
        # and the determinant of the rotation block.
        rotation_of(array)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid (4, 4) SE(3) transform: {error}") from error
    return array


def _as_pose_list(poses: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    """Return ``poses`` as a non-empty list of validated SE(3) arrays."""
    try:
        items = list(poses)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of (4, 4) transforms") from error
    if not items:
        raise ValueError(f"{name} must not be empty")
    return [_as_se3(item, f"{name}[{index}]") for index, item in enumerate(items)]


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return ``angle * axis`` for a rotation, i.e. its rotation vector.

    ``rotation_to_axis_angle`` returns the angle in ``[0, pi]`` and the
    conventional axis ``[1, 0, 0]`` at ``angle = 0``; multiplying by the angle
    turns that case into the zero vector, so it contributes nothing.
    """
    axis, angle = rotation_to_axis_angle(rotation)
    return angle * axis


def _pair_rotation_angle(rotation: np.ndarray) -> float:
    """Return the rotation angle of ``rotation`` in ``[0, pi]`` (``arccos`` form)."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


# ---------------------------------------------------------------------------
# Building relative motions
# ---------------------------------------------------------------------------


def relative_motion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the relative motion ``inv(first) @ second``.

    With ``first = T_i`` and ``second = T_j`` this is the motion of frame ``j``
    expressed in frame ``i``, which is what ``A`` and ``B`` are in ``A X = X B``.
    """
    first_pose = _as_se3(first, "first")
    second_pose = _as_se3(second, "second")
    return inverse_se3(first_pose) @ second_pose


def build_motion_pairs(
    hand_poses: Sequence[np.ndarray], camera_poses: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build ``(A_list, B_list)`` from paired end-effector and camera poses.

    ``hand_poses[i]`` and ``camera_poses[i]`` are the ``gripperframe`` and
    ``wrist_cam`` poses of the same arm configuration, both in the robot base
    frame. All unordered pairs ``i < j`` are used, in lexicographic order, with::

        A = inv(hand_poses[i]) @ hand_poses[j]
        B = inv(camera_poses[i]) @ camera_poses[j]

    Returns two lists of the same length; pair ``k`` is ``(A_list[k], B_list[k])``.
    """
    hand = _as_pose_list(hand_poses, "hand_poses")
    camera = _as_pose_list(camera_poses, "camera_poses")
    if len(hand) != len(camera):
        raise ValueError(
            "hand_poses and camera_poses must have the same length, "
            f"got {len(hand)} and {len(camera)}"
        )
    if len(hand) < 2:
        raise ValueError("at least two pose pairs are needed to form a relative motion")

    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for i in range(len(hand)):
        for j in range(i + 1, len(hand)):
            a_list.append(inverse_se3(hand[i]) @ hand[j])
            b_list.append(inverse_se3(camera[i]) @ camera[j])
    return a_list, b_list


# ---------------------------------------------------------------------------
# AX = XB
# ---------------------------------------------------------------------------


def solve_ax_xb(A_list: Sequence[np.ndarray], B_list: Sequence[np.ndarray]) -> np.ndarray:
    """Solve ``A_i X = X B_i`` for the fixed rigid transform ``X``.

    Parameters
    ----------
    A_list, B_list
        Relative motions of two frames that are rigidly connected by ``X``, as
        ``(4, 4)`` SE(3) arrays. They must have the same length.

    Returns
    -------
    numpy.ndarray
        The ``(4, 4)`` float64 ``X = pose(R_X, t_X)``.

    Raises
    ------
    ValueError
        If the inputs are not valid SE(3) transforms, are empty, have different
        lengths, contain fewer than :data:`MIN_USABLE_PAIRS` relative rotations in
        the usable range, or only contain relative rotations about a single axis.

    Notes
    -----
    The rotation is recovered from the rotation-vector correspondence
    ``log(R_A) = R_X log(R_B)`` by orthogonal Procrustes (SVD), and the
    translation from the stacked linear system of L02 §10.3 by least squares.
    See the module docstring for the derivation, the pair filter, and the note on
    notebook cell 92.
    """
    a_list = _as_pose_list(A_list, "A_list")
    b_list = _as_pose_list(B_list, "B_list")
    if len(a_list) != len(b_list):
        raise ValueError(
            f"A_list and B_list must have the same length, got {len(a_list)} and {len(b_list)}"
        )

    # --- rotation: log(R_A) = R_X log(R_B) -----------------------------------
    h_matrix = np.zeros((3, 3), dtype=_DTYPE)
    usable_pairs = 0
    for a_transform, b_transform in zip(a_list, b_list, strict=True):
        rotation_a = a_transform[:3, :3]
        angle = _pair_rotation_angle(rotation_a)
        if not MIN_RELATIVE_ROTATION_ANGLE <= angle <= MAX_RELATIVE_ROTATION_ANGLE:
            continue
        h_matrix += np.outer(_rotation_vector(rotation_a), _rotation_vector(b_transform[:3, :3]))
        usable_pairs += 1

    if usable_pairs < MIN_USABLE_PAIRS:
        raise ValueError(
            f"need at least {MIN_USABLE_PAIRS} pairs with a relative rotation in "
            f"[{MIN_RELATIVE_ROTATION_ANGLE}, {MAX_RELATIVE_ROTATION_ANGLE}] rad, "
            f"got {usable_pairs}"
        )

    u_matrix, singular_values, vt_matrix = np.linalg.svd(h_matrix)
    rotation_x = u_matrix @ vt_matrix
    if np.linalg.det(rotation_x) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation_x = u_matrix @ vt_matrix

    if singular_values[0] <= 0.0 or singular_values[1] < _RANK_TOL * singular_values[0]:
        raise ValueError(
            "relative rotations are too collinear to determine the rotation: "
            f"singular values {singular_values}"
        )

    # --- translation: (R_A - I) t_X = R_X t_B - t_A --------------------------
    row_blocks = []
    right_blocks = []
    for a_transform, b_transform in zip(a_list, b_list, strict=True):
        row_blocks.append(a_transform[:3, :3] - np.eye(3, dtype=_DTYPE))
        right_blocks.append(rotation_x @ b_transform[:3, 3] - a_transform[:3, 3])
    rows = np.concatenate(row_blocks, axis=0)
    right_hand_side = np.concatenate(right_blocks, axis=0)
    translation_x, _, _, _ = np.linalg.lstsq(rows, right_hand_side, rcond=None)

    transform = pose(rotation_x, translation_x)
    # ``pose`` already validates, but keep the float64 contract explicit.
    return np.asarray(transform, dtype=_DTYPE)
