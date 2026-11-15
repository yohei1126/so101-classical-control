"""Tests for eye-in-hand hand-eye calibration ``A X = X B``.

What is checked
---------------
1. A synthetic noiseless problem in the style of L02 §10.1: a known ``X_gt`` is
   used to generate ``B_i`` and ``A_i = X_gt B_i inv(X_gt)``, and the solver must
   recover ``X_gt``. This proves the solver, not MuJoCo.
2. The Issue #8 acceptance test: pose pairs are collected from the MuJoCo backend
   in many arm configurations, the transform ``X`` is solved from the relative
   motions alone, and the result is compared with the ground-truth
   ``X_gt = inv(T_gripperframe) @ T_wrist_cam`` measured at a reference
   configuration. The same ``X_gt`` is required to be constant across all
   collected configurations, which is the rigidity claim of Issue #7.
3. Helper behaviour, the pair filter, and invalid-input handling.

Tolerances
----------
- **Synthetic noiseless test**: ``1e-12`` for the translation norm and ``1e-12``
  rad for the rotation. The problem is exactly consistent, so double-precision
  arithmetic is expected to reach machine precision; no MuJoCo is involved.
- **MuJoCo ground-truth test**: ``1e-6`` m for the translation (Euclidean norm and
  per-axis ``atol``, ``rtol=0``) and ``1e-6`` rad for the rotation, the same
  bounds as ``tests/test_dh_fk.py`` and ``tests/test_sim_cameras.py``. The
  residuals are printed by the test, as Issue #8 asks.
- The rotation error is the angle of ``R_est.T @ R_gt``. Two evaluations of the
  same quantity are used:
  - :func:`_rotation_error` is the ``arccos(clip((trace - 1) / 2, -1, 1))`` form
    used by ``tests/test_dh_fk.py`` and ``tests/test_sim_cameras.py``. It is the
    pass/fail metric for the MuJoCo test.
  - :func:`_precise_rotation_error` computes the same angle as
    ``arctan2(|skew| / 2, (trace - 1) / 2)``. The ``arccos`` form cannot resolve
    below about ``1e-8`` rad near zero error (a ``1e-16`` error in the trace
    becomes a ``sqrt(2e-16)`` angle), so the ``1e-12`` rad synthetic tolerance is
    checked with the ``arctan2`` form and the MuJoCo test asserts both. This is
    not a looser tolerance; it is the same angle, evaluated where it is
    well-conditioned.
- The Frobenius norm ``||R_est - R_gt||`` is also printed for both tests, but it
  is never the pass/fail metric.

Data and conventions
--------------------
``X`` is the fixed transform from the wrist-camera frame to the end-effector
frame, ``X = inv(T_gripperframe) @ T_wrist_cam``, with both poses in the robot
base frame and in the **MuJoCo camera convention** (the camera looks along its
own ``-Z``). For configurations ``i`` and ``j``::

    A = inv(T_gripperframe_i) @ T_gripperframe_j
    B = inv(T_wrist_cam_i)    @ T_wrist_cam_j

so ``A X = X B``. ``B`` is built from camera poses rather than from the static
``desk_marker`` site. The marker form ``T_cam_marker_i @ inv(T_cam_marker_j)``
with ``T_cam_marker = inv(T_wrist_cam) @ T_desk_marker`` is the same matrix
(note the product order), and camera poses do not depend on the marker being
inside the frustum.

The MuJoCo test uses :data:`CONFIGURATION_COUNT` = 24 configurations: ``q = 0``,
the documented :data:`VISIBILITY_POSE` of ``tests/test_sim_cameras.py``, the
mid-range configuration of the joint ranges, and 21 seeded random configurations
inside the joint stops (:data:`SEED`, :data:`LIMIT_MARGIN`). It forms every
unordered pair, ``24 * 23 / 2`` = :data:`PAIR_COUNT` = 276 of them, and prints how
many survive the solver's documented rotation filter.

No OpenCV, no rendering, no image processing, no marker detection, and no added
noise: the noise sweep is a separate issue, and the eye-to-hand case of the
desk-fixed ``top_cam`` is another.
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np
import pytest

from so101_classical_control.backend import JOINT_NAMES
from so101_classical_control.calibration import (
    MAX_RELATIVE_ROTATION_ANGLE,
    MIN_RELATIVE_ROTATION_ANGLE,
    MIN_USABLE_PAIRS,
    build_motion_pairs,
    relative_motion,
    solve_ax_xb,
)
from so101_classical_control.kinematics import (
    inverse_se3,
    pose,
    quaternion_to_rotation,
    rotation_of,
    rz,
    translation_of,
)
from so101_classical_control.sim import SimBackend
from so101_classical_control.sim.scene import scene_xml_path

SEED = 20261114
#: Joint-limit margin used to sample configurations inside the joint stops.
LIMIT_MARGIN = 1e-2  # radians
#: Number of MuJoCo arm configurations behind the acceptance test.
CONFIGURATION_COUNT = 24
#: Unordered pairs formed from :data:`CONFIGURATION_COUNT` configurations.
PAIR_COUNT = CONFIGURATION_COUNT * (CONFIGURATION_COUNT - 1) // 2
#: Number of synthetic motion pairs in the noiseless solver test.
SYNTHETIC_PAIRS = 64

TRANSLATION_TOL = 1e-6  # metres
ROTATION_TOL = 1e-6  # radians
SYNTHETIC_TRANSLATION_TOL = 1e-12  # metres
SYNTHETIC_ROTATION_TOL = 1e-12  # radians
#: Random relative rotations must have a nonzero angle for ``B`` to be a motion.
SYNTHETIC_MOTION_RADIUS = 0.15  # metres

#: Documented arm configuration, identical to ``tests/test_sim_cameras.py``.
#: Joint values in :data:`JOINT_NAMES` order, radians.
VISIBILITY_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.1,
    "elbow_flex": 0.0,
    "wrist_flex": 1.4,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    """The compiled scene, used for the joint ranges."""
    return mujoco.MjModel.from_xml_path(str(scene_xml_path()))


def _joint_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, name
    low, high = model.jnt_range[joint_id]
    return float(low), float(high)


def _pose_arm(backend: SimBackend, targets: dict[str, float]) -> dict[str, float]:
    """Command ``targets``, settle, and return the joint positions actually read."""
    backend.command_joint_targets(targets)
    backend.step(1.0)
    return backend.read_joint_positions()


def _configurations(model: mujoco.MjModel, rng: np.random.Generator) -> list[dict[str, float]]:
    """Return the documented MuJoCo configurations of the acceptance test."""
    mid_range = {name: sum(_joint_range(model, name)) / 2.0 for name in JOINT_NAMES[:5]}
    configurations: list[dict[str, float]] = [
        dict.fromkeys(JOINT_NAMES, 0.0),
        dict(VISIBILITY_POSE),
        {**mid_range, "gripper": 0.0},
    ]
    for _ in range(CONFIGURATION_COUNT - len(configurations)):
        configurations.append(
            {
                name: float(
                    rng.uniform(
                        _joint_range(model, name)[0] + LIMIT_MARGIN,
                        _joint_range(model, name)[1] - LIMIT_MARGIN,
                    )
                )
                for name in JOINT_NAMES
            }
        )
    return configurations


def _rotation_error(r_left: np.ndarray, r_right: np.ndarray) -> float:
    """Return the angle in radians of ``r_left.T @ r_right`` (``arccos`` form)."""
    cosine = np.clip((np.trace(r_left.T @ r_right) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _precise_rotation_error(r_left: np.ndarray, r_right: np.ndarray) -> float:
    """Return the same angle as :func:`_rotation_error`, accurate near zero.

    ``arccos`` loses about half the digits at small angles; ``arctan2`` of the
    skew and trace parts stays well-conditioned.
    """
    relative = r_left.T @ r_right
    sin_part = 0.5 * float(
        np.linalg.norm(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
    )
    cos_part = 0.5 * (float(np.trace(relative)) - 1.0)
    return float(np.arctan2(sin_part, cos_part))


def _relative_rotation_angle(transform: np.ndarray) -> float:
    """Return the rotation angle of ``transform`` in ``[0, pi]``."""
    return _rotation_error(rotation_of(transform), np.eye(3))


def _assert_se3(transform: np.ndarray, label: str) -> None:
    assert isinstance(transform, np.ndarray), label
    assert transform.shape == (4, 4), label
    assert transform.dtype == np.float64, label
    np.testing.assert_array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12), label


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    """Return a random SE(3) transform with a uniformly random rotation."""
    rotation = quaternion_to_rotation(rng.normal(size=4))
    translation = rng.uniform(-SYNTHETIC_MOTION_RADIUS, SYNTHETIC_MOTION_RADIUS, 3)
    return pose(rotation, translation)


def _synthetic_problem() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return ``(X_gt, A_list, B_list)`` for an exactly consistent problem."""
    rng = np.random.default_rng(SEED)
    # A wrist-camera-sized X: a large rotation and centimetre-scale translation.
    x_gt = pose(
        quaternion_to_rotation(np.array([0.62, 0.35, -0.55, 0.42])),
        np.array([-0.045, 0.055, -0.015]),
    )
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for _ in range(SYNTHETIC_PAIRS):
        b_transform = _random_transform(rng)
        a_list.append(x_gt @ b_transform @ inverse_se3(x_gt))
        b_list.append(b_transform)
    return x_gt, a_list, b_list


# ---------------------------------------------------------------------------
# Synthetic noiseless problem (L02 §10.1 style)
# ---------------------------------------------------------------------------


def test_synthetic_noiseless_data_is_exactly_consistent() -> None:
    """``A X = X B`` must hold for the generated pairs before the solver runs."""
    x_gt, a_list, b_list = _synthetic_problem()
    for index, (a_transform, b_transform) in enumerate(
        zip(a_list, b_list, strict=True)
    ):
        np.testing.assert_allclose(
            a_transform @ x_gt,
            x_gt @ b_transform,
            atol=1e-15,
            rtol=0.0,
            err_msg=f"synthetic pair {index}",
        )


def test_synthetic_noiseless_problem_recovers_ground_truth() -> None:
    """The solver must recover a known ``X`` to machine precision."""
    x_gt, a_list, b_list = _synthetic_problem()
    estimated = solve_ax_xb(a_list, b_list)
    _assert_se3(estimated, "estimated X")

    translation_error = float(np.linalg.norm(translation_of(estimated) - translation_of(x_gt)))
    rotation_error = _precise_rotation_error(rotation_of(estimated), rotation_of(x_gt))
    frobenius_error = float(np.linalg.norm(rotation_of(estimated) - rotation_of(x_gt)))
    print(
        f"\nsynthetic: {len(a_list)} pairs, rotation error {rotation_error:.3e} rad, "
        f"translation error {translation_error:.3e} m, "
        f"rotation Frobenius {frobenius_error:.3e}"
    )

    np.testing.assert_allclose(
        translation_of(estimated),
        translation_of(x_gt),
        atol=SYNTHETIC_TRANSLATION_TOL,
        rtol=0.0,
    )
    assert rotation_error <= SYNTHETIC_ROTATION_TOL, f"{rotation_error:.3e} rad"


def test_zero_rotation_pair_contributes_nothing() -> None:
    """A pair with no relative rotation must not bias the rotation estimate.

    ``rotation_to_axis_angle`` returns the conventional axis ``[1, 0, 0]`` with
    ``angle = 0``; the solver must use the zero rotation vector for that case, so
    adding such a pair leaves the recovered ``X`` at machine precision.
    """
    x_gt, a_list, b_list = _synthetic_problem()
    a_list = a_list[:8]
    b_list = b_list[:8]
    a_list.append(np.eye(4))
    b_list.append(np.eye(4))

    estimated = solve_ax_xb(a_list, b_list)
    translation_error = float(np.linalg.norm(translation_of(estimated) - translation_of(x_gt)))
    rotation_error = _precise_rotation_error(rotation_of(estimated), rotation_of(x_gt))
    assert translation_error <= SYNTHETIC_TRANSLATION_TOL, f"{translation_error:.3e} m"
    assert rotation_error <= SYNTHETIC_ROTATION_TOL, f"{rotation_error:.3e} rad"


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------


def test_relative_motion_composes_transforms() -> None:
    rng = np.random.default_rng(SEED)
    first = _random_transform(rng)
    second = _random_transform(rng)
    np.testing.assert_allclose(
        relative_motion(first, second), inverse_se3(first) @ second, atol=1e-15, rtol=0.0
    )


def test_build_motion_pairs_returns_all_unordered_pairs() -> None:
    rng = np.random.default_rng(SEED)
    hand_poses = [_random_transform(rng) for _ in range(5)]
    camera_poses = [hand @ _random_transform(rng) for hand in hand_poses]

    a_list, b_list = build_motion_pairs(hand_poses, camera_poses)

    assert len(a_list) == len(b_list) == 5 * 4 // 2
    for a_transform, b_transform in zip(a_list, b_list, strict=True):
        _assert_se3(a_transform, "A")
        _assert_se3(b_transform, "B")
    # The pairs are lexicographic: (0, 1), (0, 2), ..., (3, 4).
    index_pairs = list(itertools.combinations(range(5), 2))
    assert len(index_pairs) == len(a_list)
    np.testing.assert_allclose(
        a_list[0], inverse_se3(hand_poses[0]) @ hand_poses[1], atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        b_list[0], inverse_se3(camera_poses[0]) @ camera_poses[1], atol=1e-15, rtol=0.0
    )


# ---------------------------------------------------------------------------
# MuJoCo ground-truth acceptance test (Issue #8)
# ---------------------------------------------------------------------------


def test_eye_in_hand_ax_xb_matches_mujoco_ground_truth(model: mujoco.MjModel) -> None:
    rng = np.random.default_rng(SEED)
    configurations = _configurations(model, rng)
    assert len(configurations) == CONFIGURATION_COUNT

    with SimBackend() as backend:
        hand_poses: list[np.ndarray] = []
        camera_poses: list[np.ndarray] = []
        for targets in configurations:
            _pose_arm(backend, targets)
            hand_poses.append(backend.get_site_pose("gripperframe"))
            camera_poses.append(backend.get_camera_pose("wrist_cam"))

    # Ground truth: the same X at every configuration, measured at the first one.
    ground_truth = inverse_se3(hand_poses[0]) @ camera_poses[0]
    _assert_se3(ground_truth, "ground-truth X")
    for index, (hand_pose, camera_pose) in enumerate(
        zip(hand_poses, camera_poses, strict=True)
    ):
        measured = inverse_se3(hand_pose) @ camera_pose
        np.testing.assert_allclose(
            translation_of(measured),
            translation_of(ground_truth),
            atol=TRANSLATION_TOL,
            rtol=0.0,
            err_msg=f"X translation drifted at configuration {index}",
        )
        error = _rotation_error(rotation_of(measured), rotation_of(ground_truth))
        assert error <= ROTATION_TOL, f"configuration {index}: X rotation drifted ({error:.3e} rad)"

    # Calibrate from relative motions only.
    a_list, b_list = build_motion_pairs(hand_poses, camera_poses)
    assert len(a_list) == PAIR_COUNT

    usable_pairs = sum(
        1
        for a_transform in a_list
        if MIN_RELATIVE_ROTATION_ANGLE
        <= _relative_rotation_angle(a_transform)
        <= MAX_RELATIVE_ROTATION_ANGLE
    )
    assert usable_pairs >= MIN_USABLE_PAIRS
    assert usable_pairs > len(a_list) // 2, (
        "most pairs should have a mid-range relative rotation; "
        f"only {usable_pairs} of {len(a_list)} do"
    )

    estimated = solve_ax_xb(a_list, b_list)
    _assert_se3(estimated, "estimated X")

    translation_offset = translation_of(estimated) - translation_of(ground_truth)
    translation_error = float(np.linalg.norm(translation_offset))
    per_axis_error = float(np.abs(translation_offset).max())
    arccos_rotation_error = _rotation_error(rotation_of(estimated), rotation_of(ground_truth))
    rotation_error = _precise_rotation_error(rotation_of(estimated), rotation_of(ground_truth))
    frobenius_error = float(np.linalg.norm(rotation_of(estimated) - rotation_of(ground_truth)))
    print(
        f"\nmujoco: {len(configurations)} configurations, {len(a_list)} pairs "
        f"({usable_pairs} usable), rotation error {rotation_error:.3e} rad, "
        f"arccos {arccos_rotation_error:.3e} rad, translation error "
        f"{translation_error:.3e} m, max per-axis {per_axis_error:.3e} m, "
        f"rotation Frobenius {frobenius_error:.3e}"
    )

    np.testing.assert_allclose(
        translation_of(estimated),
        translation_of(ground_truth),
        atol=TRANSLATION_TOL,
        rtol=0.0,
        err_msg="recovered X translation differs from the MuJoCo ground truth",
    )
    assert rotation_error <= ROTATION_TOL, (
        f"recovered X rotation error {rotation_error:.3e} rad "
        f"(arccos form {arccos_rotation_error:.3e} rad)"
    )
    assert arccos_rotation_error <= ROTATION_TOL, f"{arccos_rotation_error:.3e} rad"


# ---------------------------------------------------------------------------
# Invalid input, unusable pairs
# ---------------------------------------------------------------------------


def test_fewer_than_minimum_usable_pairs_raises() -> None:
    # Translation-only motion: the rotation stage has no usable pair at all.
    a_list = [pose(np.eye(3), np.array([0.01 * index, 0.0, 0.0])) for index in range(1, 6)]
    b_list = [pose(np.eye(3), np.array([0.0, 0.01 * index, 0.0])) for index in range(1, 6)]
    with pytest.raises(ValueError, match="relative rotation"):
        solve_ax_xb(a_list, b_list)


def test_small_relative_rotations_are_not_usable() -> None:
    # Rotations just below the documented threshold must be filtered out.
    angle = 0.5 * MIN_RELATIVE_ROTATION_ANGLE
    a_list = [pose(rz(angle), np.array([0.01, 0.0, 0.0])) for _ in range(5)]
    b_list = [pose(rz(angle), np.array([0.0, 0.01, 0.0])) for _ in range(5)]
    with pytest.raises(ValueError, match="relative rotation"):
        solve_ax_xb(a_list, b_list)


def test_collinear_relative_rotations_raise() -> None:
    """Rotations about a single axis cannot determine the full rotation."""
    x_gt = pose(rz(0.7), np.array([0.02, 0.03, 0.04]))
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for angle in np.linspace(0.2, 2.5, 8):
        b_transform = pose(rz(float(angle)), np.array([0.01, 0.02, 0.03]))
        a_list.append(x_gt @ b_transform @ inverse_se3(x_gt))
        b_list.append(b_transform)
    with pytest.raises(ValueError, match="collinear"):
        solve_ax_xb(a_list, b_list)


@pytest.mark.parametrize(
    "bad_transform",
    [
        np.eye(3),
        np.eye(5),
        np.zeros((4, 4)),
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, np.nan], [0.0, 0.0, 0.0, 1.0]],
        "not a matrix",
    ],
)
def test_invalid_transform_raises(bad_transform) -> None:
    with pytest.raises(ValueError):
        solve_ax_xb([np.eye(4), np.eye(4), np.eye(4)], [np.eye(4), np.eye(4), bad_transform])
    with pytest.raises(ValueError):
        solve_ax_xb([np.eye(4), np.eye(4), bad_transform], [np.eye(4), np.eye(4), np.eye(4)])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        solve_ax_xb([np.eye(4), np.eye(4)], [np.eye(4)])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        solve_ax_xb([], [])
