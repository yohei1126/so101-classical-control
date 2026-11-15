"""Tests for camera-to-base (eye-to-hand) calibration of the desk-fixed camera.

What is checked
---------------
1. Synthetic noiseless checks, run **before** anything touches MuJoCo, in the
   spirit of L02 §10.1 (pick a known ``X``, generate consistent data, recover it):
   - the closed-loop transform equation
     ``T_base_cam = T_base_marker @ inv(T_cam_marker)`` recovers a known
     ``X_gt``;
   - the fixed-frame pairing ``A = T_base_marker_j @ inv(T_base_marker_i)``,
     ``B = T_cam_marker_j @ inv(T_cam_marker_i)`` recovers the same ``X_gt``
     through the existing ``solve_ax_xb``;
   - the eye-in-hand pairing of ``build_motion_pairs`` /
     ``relative_motion`` (``inv(T_i) @ T_j``) does **not** recover ``X_gt`` on
     the same data, which is why this issue does not reuse it.
2. The Issue #9 acceptance test: marker poses are collected from the MuJoCo
   backend in many arm configurations, ``T_base_cam`` is recovered from the
   closed-loop equation (every configuration) and from ``solve_ax_xb`` (all
   unordered pairs), and both are compared with
   ``get_camera_pose("top_cam")``. The test also re-checks two Issue #7 claims:
   ``top_cam`` is static, and ``inv(T_gripperframe) @ T_ee_marker`` is constant
   across the collected configurations.
3. Helper behaviour and invalid-input handling.

Tolerances
----------
- **Synthetic noiseless checks**: ``1e-12`` m for the translation norm and
  ``1e-12`` rad for the rotation. The data are exactly consistent, so double
  precision is expected to reach machine precision; no MuJoCo is involved. The
  rotation is evaluated with the well-conditioned ``arctan2`` form (see below).
- **MuJoCo ground-truth test**: ``1e-6`` m for the translation (Euclidean norm)
  and ``1e-6`` rad for the rotation, the same bounds as ``tests/test_dh_fk.py``,
  ``tests/test_sim_cameras.py``, and ``tests/test_handeye_axxb.py``. The
  recovered-vs-ground-truth residuals are printed by the test, as Issue #9 asks.
- The rotation error is the angle of ``R_est.T @ R_gt``. Two evaluations of the
  same quantity are used: :func:`_rotation_error` is the
  ``arccos(clip((trace - 1) / 2, -1, 1))`` form of the other tests and is the
  pass/fail metric for the MuJoCo test (asserted together with the precise
  form); :func:`_precise_rotation_error` computes the same angle as
  ``arctan2(|skew| / 2, (trace - 1) / 2)``, which stays well-conditioned near
  zero and is what the ``1e-12`` rad synthetic bound is checked with.
- The Frobenius norm ``||R_est - R_gt||`` is printed, but it is never the
  pass/fail metric.

Data and conventions
--------------------
``T_base_cam`` is the camera-to-base transform, i.e. the ground-truth pose
``get_camera_pose("top_cam")``, in the **MuJoCo camera convention** (the camera
looks along its own ``-Z``). ``T_base_marker`` is ``get_site_pose("ee_marker")``.
``T_cam_marker`` is formed algebraically as
``inv(T_base_cam_gt) @ T_base_marker``: nothing is detected in an image, and no
rendering is used.

The MuJoCo test uses :data:`CONFIGURATION_COUNT` = 24 configurations: ``q = 0``,
the documented :data:`VISIBILITY_POSE` of ``tests/test_sim_cameras.py``, the
mid-range configuration of the joint ranges, and 21 seeded random configurations
inside the joint stops (:data:`SEED`, :data:`LIMIT_MARGIN`). Its pairs are the
``24 * 23 / 2`` = :data:`PAIR_COUNT` = 276 unordered pairs; the test prints how
many survive the solver's documented rotation filter.

No OpenCV, no intrinsic matrix, no marker detection, and no added noise: the
noise sweep is Issue #10 and image-based detection is Issue #20.
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
    build_fixed_frame_motion_pairs,
    build_motion_pairs,
    camera_to_base,
    camera_to_base_per_pose,
    solve_ax_xb,
)
from so101_classical_control.kinematics import (
    inverse_se3,
    pose,
    quaternion_to_rotation,
    rotation_of,
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
#: Number of synthetic configurations in the noiseless checks.
SYNTHETIC_CONFIGURATIONS = 8
#: Spread of the synthetic marker positions, in metres.
SYNTHETIC_POSE_RADIUS = 0.20

TRANSLATION_TOL = 1e-6  # metres
ROTATION_TOL = 1e-6  # radians
SYNTHETIC_TRANSLATION_TOL = 1e-12  # metres
SYNTHETIC_ROTATION_TOL = 1e-12  # radians
#: The eye-in-hand pairing must miss the camera-to-base transform by far more
#: than this on the synthetic data, which is why it is not used here.
WRONG_PAIRING_TOL = 1e-2  # radians

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


def _translation_error(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``||t_left - t_right||`` in metres."""
    return float(np.linalg.norm(translation_of(left) - translation_of(right)))


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
    translation = rng.uniform(-SYNTHETIC_POSE_RADIUS, SYNTHETIC_POSE_RADIUS, 3)
    return pose(rotation, translation)


def _synthetic_problem() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return ``(X_gt, base_markers, marker_in_camera_poses)`` for exact data.

    ``X_gt`` is desk-camera sized: a large tilt and a translation of about
    0.7 m, the order of ``top_cam``'s pose. The marker poses are arbitrary
    ground-truth-style SE(3) transforms, and
    ``T_cam_marker = inv(X_gt) @ T_base_marker`` holds exactly by construction.
    """
    rng = np.random.default_rng(SEED)
    x_gt = pose(
        quaternion_to_rotation(np.array([0.85, 0.10, -0.45, 0.20])),
        np.array([0.15, -0.35, 0.60]),
    )
    base_markers = [np.eye(4, dtype=np.float64)]
    base_markers.extend(
        _random_transform(rng) for _ in range(SYNTHETIC_CONFIGURATIONS - 1)
    )
    marker_in_camera_poses = [inverse_se3(x_gt) @ base_marker for base_marker in base_markers]
    return x_gt, base_markers, marker_in_camera_poses


# ---------------------------------------------------------------------------
# Synthetic noiseless checks (run before the MuJoCo test in file order)
# ---------------------------------------------------------------------------


def test_synthetic_closed_loop_data_is_exactly_consistent() -> None:
    """``T_base_cam @ T_cam_marker = T_base_marker`` must hold before recovery."""
    x_gt, base_markers, marker_in_camera_poses = _synthetic_problem()
    for index, (base_marker, marker_in_camera) in enumerate(
        zip(base_markers, marker_in_camera_poses, strict=True)
    ):
        np.testing.assert_allclose(
            x_gt @ marker_in_camera,
            base_marker,
            atol=1e-15,
            rtol=0.0,
            err_msg=f"synthetic configuration {index}",
        )


def test_synthetic_closed_loop_recovers_known_camera_to_base() -> None:
    """``T_base_marker @ inv(T_cam_marker)`` must recover a known ``X_gt``."""
    x_gt, base_markers, marker_in_camera_poses = _synthetic_problem()

    estimates = camera_to_base_per_pose(base_markers, marker_in_camera_poses)
    assert len(estimates) == SYNTHETIC_CONFIGURATIONS

    worst_rotation_error = 0.0
    worst_translation_error = 0.0
    for index, (estimate, base_marker, marker_in_camera) in enumerate(
        zip(estimates, base_markers, marker_in_camera_poses, strict=True)
    ):
        _assert_se3(estimate, f"closed-loop estimate {index}")
        np.testing.assert_allclose(
            camera_to_base(base_marker, marker_in_camera),
            estimate,
            atol=1e-15,
            rtol=0.0,
            err_msg=f"per-pose helpers disagree at configuration {index}",
        )
        rotation_error = _precise_rotation_error(rotation_of(estimate), rotation_of(x_gt))
        translation_error = _translation_error(estimate, x_gt)
        worst_rotation_error = max(worst_rotation_error, rotation_error)
        worst_translation_error = max(worst_translation_error, translation_error)

        np.testing.assert_allclose(
            translation_of(estimate),
            translation_of(x_gt),
            atol=SYNTHETIC_TRANSLATION_TOL,
            rtol=0.0,
            err_msg=f"closed-loop translation at configuration {index}",
        )
        assert rotation_error <= SYNTHETIC_ROTATION_TOL, (
            f"configuration {index}: closed-loop rotation error {rotation_error:.3e} rad"
        )

    print(
        f"\nsynthetic closed loop: {len(estimates)} configurations, worst rotation error "
        f"{worst_rotation_error:.3e} rad, worst translation error "
        f"{worst_translation_error:.3e} m"
    )


def test_synthetic_fixed_frame_pairs_satisfy_ax_xb() -> None:
    """``A X = X B`` must hold for the ``T_j @ inv(T_i)`` pairing."""
    x_gt, base_markers, marker_in_camera_poses = _synthetic_problem()
    a_list, b_list = build_fixed_frame_motion_pairs(base_markers, marker_in_camera_poses)

    assert len(a_list) == len(b_list) == SYNTHETIC_CONFIGURATIONS * (
        SYNTHETIC_CONFIGURATIONS - 1
    ) // 2
    for index, (a_transform, b_transform) in enumerate(zip(a_list, b_list, strict=True)):
        _assert_se3(a_transform, "A")
        _assert_se3(b_transform, "B")
        np.testing.assert_allclose(
            a_transform @ x_gt,
            x_gt @ b_transform,
            atol=1e-15,
            rtol=0.0,
            err_msg=f"synthetic pair {index}",
        )


def test_synthetic_fixed_frame_pairs_recover_known_camera_to_base() -> None:
    """``solve_ax_xb`` on the ``T_j @ inv(T_i)`` pairing must recover ``X_gt``."""
    x_gt, base_markers, marker_in_camera_poses = _synthetic_problem()
    a_list, b_list = build_fixed_frame_motion_pairs(base_markers, marker_in_camera_poses)

    estimated = solve_ax_xb(a_list, b_list)
    _assert_se3(estimated, "estimated camera_to_base")

    translation_error = _translation_error(estimated, x_gt)
    rotation_error = _precise_rotation_error(rotation_of(estimated), rotation_of(x_gt))
    frobenius_error = float(np.linalg.norm(rotation_of(estimated) - rotation_of(x_gt)))
    print(
        f"\nsynthetic A X = X B: {len(a_list)} pairs, rotation error {rotation_error:.3e} rad, "
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


def test_issue8_pairing_does_not_recover_camera_to_base() -> None:
    """The eye-in-hand pairing ``inv(T_i) @ T_j`` must fail on the same data.

    ``build_motion_pairs`` builds ``A = inv(T_i) @ T_j``, the eye-in-hand
    convention of Issue #8. For camera-to-base calibration the fixed-frame order
    ``T_j @ inv(T_i)`` is required, so the eye-in-hand pairing must miss both
    ``T_base_cam`` and its inverse. This is a documented negative check with the
    existing solver; no second solver is added.
    """
    x_gt, base_markers, marker_in_camera_poses = _synthetic_problem()
    a_list, b_list = build_motion_pairs(base_markers, marker_in_camera_poses)

    estimated = solve_ax_xb(a_list, b_list)
    _assert_se3(estimated, "eye-in-hand pairing estimate")

    rotation_error = _precise_rotation_error(rotation_of(estimated), rotation_of(x_gt))
    inverse_rotation_error = _precise_rotation_error(
        rotation_of(estimated), rotation_of(inverse_se3(x_gt))
    )
    print(
        f"\nsynthetic Issue #8 pairing: rotation error {rotation_error:.3e} rad vs X_gt, "
        f"{inverse_rotation_error:.3e} rad vs inv(X_gt)"
    )

    assert rotation_error > WRONG_PAIRING_TOL, (
        "the eye-in-hand pairing unexpectedly recovered T_base_cam "
        f"(rotation error {rotation_error:.3e} rad)"
    )
    assert inverse_rotation_error > WRONG_PAIRING_TOL, (
        "the eye-in-hand pairing unexpectedly recovered inv(T_base_cam) "
        f"(rotation error {inverse_rotation_error:.3e} rad)"
    )


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------


def test_camera_to_base_composes_transforms() -> None:
    """The closed-loop helper must be ``base_marker @ inv(cam_marker)``."""
    rng = np.random.default_rng(SEED)
    base_marker = _random_transform(rng)
    marker_in_camera = _random_transform(rng)

    estimated = camera_to_base(base_marker, marker_in_camera)

    _assert_se3(estimated, "camera_to_base")
    np.testing.assert_allclose(
        estimated,
        base_marker @ inverse_se3(marker_in_camera),
        atol=1e-15,
        rtol=0.0,
    )


def test_build_fixed_frame_motion_pairs_returns_all_unordered_pairs() -> None:
    """The pairs must be lexicographic and use ``T_j @ inv(T_i)``."""
    rng = np.random.default_rng(SEED)
    base_markers = [_random_transform(rng) for _ in range(5)]
    marker_in_camera_poses = [inverse_se3(_random_transform(rng)) @ b for b in base_markers]

    a_list, b_list = build_fixed_frame_motion_pairs(base_markers, marker_in_camera_poses)

    assert len(a_list) == len(b_list) == 5 * 4 // 2
    assert len(list(itertools.combinations(range(5), 2))) == len(a_list)
    np.testing.assert_allclose(
        a_list[0], base_markers[1] @ inverse_se3(base_markers[0]), atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        b_list[0],
        marker_in_camera_poses[1] @ inverse_se3(marker_in_camera_poses[0]),
        atol=1e-15,
        rtol=0.0,
    )
    # The eye-in-hand convention is a different matrix: the two must not be
    # confused when the relative rotation is nonzero.
    np.testing.assert_allclose(
        build_motion_pairs(base_markers, marker_in_camera_poses)[0][0],
        inverse_se3(base_markers[0]) @ base_markers[1],
        atol=1e-15,
        rtol=0.0,
    )
    assert _relative_rotation_angle(a_list[0]) > 1e-3


# ---------------------------------------------------------------------------
# MuJoCo ground-truth acceptance test (Issue #9)
# ---------------------------------------------------------------------------


def test_camera_to_base_matches_mujoco_ground_truth(model: mujoco.MjModel) -> None:
    rng = np.random.default_rng(SEED)
    configurations = _configurations(model, rng)
    assert len(configurations) == CONFIGURATION_COUNT

    base_markers: list[np.ndarray] = []
    marker_in_camera_poses: list[np.ndarray] = []
    camera_poses: list[np.ndarray] = []
    gripper_frames: list[np.ndarray] = []
    with SimBackend() as backend:
        for targets in configurations:
            _pose_arm(backend, targets)
            base_marker = backend.get_site_pose("ee_marker")
            camera_pose = backend.get_camera_pose("top_cam")
            base_markers.append(base_marker)
            camera_poses.append(camera_pose)
            gripper_frames.append(backend.get_site_pose("gripperframe"))
            marker_in_camera_poses.append(inverse_se3(camera_pose) @ base_marker)

    # Ground truth: the desk-fixed camera pose, measured at the first
    # configuration. Issue #7 claims it is static while the arm moves.
    ground_truth = camera_poses[0]
    worst_camera_drift_translation = 0.0
    worst_camera_drift_rotation = 0.0
    for index, camera_pose in enumerate(camera_poses):
        drift_translation = _translation_error(camera_pose, ground_truth)
        drift_rotation = _rotation_error(rotation_of(camera_pose), rotation_of(ground_truth))
        worst_camera_drift_translation = max(worst_camera_drift_translation, drift_translation)
        worst_camera_drift_rotation = max(worst_camera_drift_rotation, drift_rotation)
        assert drift_translation <= TRANSLATION_TOL, (
            f"top_cam moved {drift_translation:.3e} m at configuration {index}"
        )
        assert drift_rotation <= ROTATION_TOL, (
            f"top_cam rotated {drift_rotation:.3e} rad at configuration {index}"
        )

    # Issue #7: ee_marker is rigid with respect to gripperframe.
    rigid_marker = inverse_se3(gripper_frames[0]) @ base_markers[0]
    worst_rigidity_translation = 0.0
    worst_rigidity_rotation = 0.0
    for index, (gripper_frame, base_marker) in enumerate(
        zip(gripper_frames, base_markers, strict=True)
    ):
        measured = inverse_se3(gripper_frame) @ base_marker
        rigidity_translation = _translation_error(measured, rigid_marker)
        rigidity_rotation = _rotation_error(rotation_of(measured), rotation_of(rigid_marker))
        worst_rigidity_translation = max(worst_rigidity_translation, rigidity_translation)
        worst_rigidity_rotation = max(worst_rigidity_rotation, rigidity_rotation)
        assert rigidity_translation <= TRANSLATION_TOL, (
            f"ee_marker moved {rigidity_translation:.3e} m relative to gripperframe "
            f"at configuration {index}"
        )
        assert rigidity_rotation <= ROTATION_TOL, (
            f"ee_marker rotated {rigidity_rotation:.3e} rad relative to gripperframe "
            f"at configuration {index}"
        )

    # Estimate 1: the closed-loop transform equation at every configuration.
    closed_loop_estimates = camera_to_base_per_pose(base_markers, marker_in_camera_poses)
    assert len(closed_loop_estimates) == CONFIGURATION_COUNT
    worst_closed_loop_translation = 0.0
    worst_closed_loop_rotation = 0.0
    for index, estimate in enumerate(closed_loop_estimates):
        _assert_se3(estimate, f"closed-loop estimate {index}")
        translation_error = _translation_error(estimate, ground_truth)
        rotation_error = _rotation_error(rotation_of(estimate), rotation_of(ground_truth))
        precise_rotation_error = _precise_rotation_error(
            rotation_of(estimate), rotation_of(ground_truth)
        )
        worst_closed_loop_translation = max(worst_closed_loop_translation, translation_error)
        worst_closed_loop_rotation = max(worst_closed_loop_rotation, precise_rotation_error)
        assert translation_error <= TRANSLATION_TOL, (
            f"configuration {index}: closed-loop translation error {translation_error:.3e} m"
        )
        assert rotation_error <= ROTATION_TOL, (
            f"configuration {index}: closed-loop rotation error {rotation_error:.3e} rad"
        )
        assert precise_rotation_error <= ROTATION_TOL, (
            f"configuration {index}: closed-loop rotation error "
            f"{precise_rotation_error:.3e} rad (arctan2 form)"
        )

    # Estimate 2: reuse solve_ax_xb on the fixed-frame pairing.
    a_list, b_list = build_fixed_frame_motion_pairs(base_markers, marker_in_camera_poses)
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

    ax_xb_estimate = solve_ax_xb(a_list, b_list)
    _assert_se3(ax_xb_estimate, "solve_ax_xb estimate")

    ax_xb_translation_offset = translation_of(ax_xb_estimate) - translation_of(ground_truth)
    ax_xb_translation_error = float(np.linalg.norm(ax_xb_translation_offset))
    ax_xb_arccos_rotation_error = _rotation_error(
        rotation_of(ax_xb_estimate), rotation_of(ground_truth)
    )
    ax_xb_rotation_error = _precise_rotation_error(
        rotation_of(ax_xb_estimate), rotation_of(ground_truth)
    )
    ax_xb_frobenius_error = float(
        np.linalg.norm(rotation_of(ax_xb_estimate) - rotation_of(ground_truth))
    )
    closed_loop_frobenius_error = float(
        np.linalg.norm(rotation_of(closed_loop_estimates[0]) - rotation_of(ground_truth))
    )

    print(
        f"\nmujoco: {len(configurations)} configurations, {len(a_list)} pairs "
        f"({usable_pairs} usable)\n"
        f"  top_cam drift: {worst_camera_drift_rotation:.3e} rad, "
        f"{worst_camera_drift_translation:.3e} m\n"
        f"  ee_marker rigidity: {worst_rigidity_rotation:.3e} rad, "
        f"{worst_rigidity_translation:.3e} m\n"
        f"  closed loop vs ground truth (worst): {worst_closed_loop_rotation:.3e} rad, "
        f"{worst_closed_loop_translation:.3e} m, "
        f"rotation Frobenius {closed_loop_frobenius_error:.3e}\n"
        f"  A X = X B vs ground truth: {ax_xb_rotation_error:.3e} rad "
        f"(arccos {ax_xb_arccos_rotation_error:.3e} rad), "
        f"{ax_xb_translation_error:.3e} m, "
        f"max per-axis {float(np.abs(ax_xb_translation_offset).max()):.3e} m, "
        f"rotation Frobenius {ax_xb_frobenius_error:.3e}"
    )

    np.testing.assert_allclose(
        translation_of(ax_xb_estimate),
        translation_of(ground_truth),
        atol=TRANSLATION_TOL,
        rtol=0.0,
        err_msg="recovered camera-to-base translation differs from the MuJoCo ground truth",
    )
    assert ax_xb_rotation_error <= ROTATION_TOL, (
        f"recovered camera-to-base rotation error {ax_xb_rotation_error:.3e} rad "
        f"(arccos form {ax_xb_arccos_rotation_error:.3e} rad)"
    )
    assert ax_xb_arccos_rotation_error <= ROTATION_TOL, f"{ax_xb_arccos_rotation_error:.3e} rad"


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


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
        camera_to_base(bad_transform, np.eye(4))
    with pytest.raises(ValueError):
        camera_to_base(np.eye(4), bad_transform)
    with pytest.raises(ValueError):
        camera_to_base_per_pose([np.eye(4), bad_transform], [np.eye(4), np.eye(4)])
    with pytest.raises(ValueError):
        build_fixed_frame_motion_pairs([np.eye(4), bad_transform], [np.eye(4), np.eye(4)])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        camera_to_base_per_pose([np.eye(4), np.eye(4)], [np.eye(4)])
    with pytest.raises(ValueError, match="same length"):
        build_fixed_frame_motion_pairs([np.eye(4), np.eye(4)], [np.eye(4)])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        camera_to_base_per_pose([], [])
    with pytest.raises(ValueError):
        build_fixed_frame_motion_pairs([], [])


def test_single_configuration_pair_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_fixed_frame_motion_pairs([np.eye(4)], [np.eye(4)])
