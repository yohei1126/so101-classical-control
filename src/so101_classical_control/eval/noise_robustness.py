"""Noise-robustness sweep for the two hand-eye calibrations.

This module measures how the eye-in-hand ``A X = X B`` calibration of Issue #8
(``calibration/hand_eye.py``) and the camera-to-base (eye-to-hand) calibration of
Issue #9 (``calibration/eye_to_hand.py``) degrade under synthetic Gaussian noise.
It only **calls** those solvers; it does not modify them, and it does not import
OpenCV, detect a marker in an image, or run inverse kinematics. The noise lives
on the L01-L02 quantities only: joint angles and ground-truth poses.

Sources
-------
- Issue [#10](https://github.com/yohei1126/so101-classical-control/issues/10):
  the noise-robustness sweep itself.
- L02 lecture notes, section "3. Hand-Eye Calibration" and §2.7 (closed-loop
  transform equation), in
  ``PhysicalAI-Applied2-202611/L02_20261119`` of
  ``yohei1126/utokyo-ai-courses``: the two calibrations swept here.
- ``docs/calibration.md``: the frames, the solvers, and the noiseless test
  tolerances. The ground-truth poses come from ``SimBackend`` exactly as in
  ``tests/test_handeye_axxb.py`` and ``tests/test_eyetohand_camerabase.py``.

Noise model
-----------
A noise level ``s`` is dimensionless. It scales three independent Gaussians:

- **joint noise**: ``s`` degrees of standard deviation per arm joint (the five
  joints of :data:`~so101_classical_control.kinematics.ARM_JOINT_NAMES`; the
  gripper DoF does not move ``gripperframe`` and is not perturbed);
- **marker rotation noise**: ``s`` degrees per axis of the measured marker pose;
- **marker translation noise**: ``MARKER_TRANSLATION_MM_PER_LEVEL * s``
  millimetres per axis of the measured marker pose.

Joint angles are **noisy encoder estimates** fed to the DH forward kinematics.
The arm physically sits at the true configuration: only the reported joint
angles are perturbed, so the end-effector pose the methods use (and therefore the
``ee_marker`` position they derive from it) is the DH estimate of the noisy
angles. The camera and marker observations are of the **true** poses — the scene
does not move when an encoder is misread — plus the marker-detection noise above
when that channel is switched on. Joint noise therefore enters only through the
robot-side estimate, and marker noise only through the measured marker pose.

A marker pose ``T`` is perturbed on the right, ``T @ pose(R_delta, t_delta)``, so
the noise is expressed in the measured frame. Level ``s = 0`` returns the input
pose unchanged.

Ground truth
------------
:func:`collect_ground_truth` drives the public backend through the same 24
configurations as the calibration tests and records the reached joint positions
plus the camera and site poses. It also checks that
:func:`~so101_classical_control.kinematics.forward_kinematics` reproduces the
recorded ``gripperframe`` pose, so the joint-noise path is anchored to MuJoCo,
and that ``top_cam`` and ``desk_marker`` are static across configurations.

Methods
-------
Four estimates are compared against ground truth on the same trial:

- ``hand_eye_axxb``: wrist eye-in-hand ``A X = X B`` (Issue #8), with the
  camera-side pose estimated from the static ``desk_marker``.
- ``camera_to_base_closed_loop_single``: one configuration of the camera-to-base
  closed-loop equation (Issue #9). A cell with grid pose count ``N`` draws its
  single configuration as ``trial % N``, i.e. within that cell's first ``N``
  configurations, so the method is never pooled across pose-count cells. Its
  ``num_poses`` column stays ``1`` and ``cell_pose_count`` records the grid ``N``.
- ``camera_to_base_closed_loop_mean``: the mean of the per-pose closed-loop
  estimates, with the rotation averaged by projection onto SO(3). This is an
  aggregation of existing estimates, not a new solver.
- ``camera_to_base_axxb``: the fixed-frame ``A X = X B`` pairing (Issue #9).

Metrics
-------
The rotation error is the angle of ``R_est.T @ R_gt``, evaluated with the
well-conditioned ``arctan2`` form of the calibration tests (``arccos`` loses
about half the digits near zero), and reported in degrees. The translation error
is the Euclidean norm of the translation difference, in millimetres.

Reproducibility
---------------
Every ``(noise source, level, pose count, trial)`` cell gets its own generator,
seeded with stable tokens rather than enumeration indices::

    np.random.default_rng(
        [seed, SOURCE_TOKENS[source], round(level * 1000), cell_pose_count, trial]
    )

with :data:`SOURCE_TOKENS` mapping ``none``/``joint``/``marker``/``both`` to
``0``/``1``/``2``/``3``. A cell's rows are therefore identical whether it is run
alone or as part of a larger grid, so a subset of the grid can be regenerated
and compared with the committed summary.

Outputs
-------
``results/noise_robustness_trials.csv`` (raw, gitignored),
``results/noise_robustness_summary.csv`` (committed, small), and
``results/noise_robustness.png`` (committed). See ``results/README.md`` for the
policy. Run with::

    uv run python -m so101_classical_control.eval.noise_robustness \
        --output-dir results --trials 50 --seed 20261118

``--quick`` without ``--output-dir`` writes to ``results/quick/``, which the
``results/*`` gitignore rule keeps out of version control.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np

from ..backend import JOINT_NAMES
from ..calibration import (
    build_fixed_frame_motion_pairs,
    build_motion_pairs,
    camera_to_base,
    camera_to_base_per_pose,
    solve_ax_xb,
)
from ..kinematics import (
    ARM_JOINT_NAMES,
    axis_angle_to_rotation,
    forward_kinematics,
    inverse_se3,
    pose,
)
from ..sim import SimBackend
from ..sim.scene import scene_xml_path

__all__ = [
    "CONFIGURATION_COUNT",
    "CONFIGURATION_SEED",
    "GroundTruth",
    "LIMIT_MARGIN",
    "MARKER_TRANSLATION_MM_PER_LEVEL",
    "METHODS",
    "NOISE_LEVELS",
    "NOISE_SOURCES",
    "POSE_COUNTS",
    "QUICK_OUTPUT_DIR",
    "SEED",
    "SOURCE_TOKENS",
    "SUMMARY_CSV_HEADER",
    "SweepResult",
    "TRIALS",
    "TRIAL_CSV_HEADER",
    "VISIBILITY_POSE",
    "average_poses",
    "collect_ground_truth",
    "evaluate_trial",
    "main",
    "perturb_joints",
    "perturb_pose",
    "resolve_output_dir",
    "rotation_error_deg",
    "run",
    "translation_error_mm",
]

#: Default noise-sweep seed. Stated in ``results/README.md`` and the docs.
SEED = 20261118
#: Trials per grid cell of the committed sweep.
TRIALS = 50
#: Number of MuJoCo configurations collected as ground truth.
CONFIGURATION_COUNT = 24
#: Joint-limit margin used to sample configurations inside the joint stops.
LIMIT_MARGIN = 1e-2  # radians
#: Seed of the ground-truth configurations, matching the calibration tests.
CONFIGURATION_SEED = 20261114
#: The sweep uses the first ``N`` configurations of the ground truth.
POSE_COUNTS: Final[tuple[int, ...]] = (6, 12, 24)
#: Dimensionless noise levels ``s``.
NOISE_LEVELS: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
#: Which synthetic noise is switched on; the other one stays at zero.
NOISE_SOURCES: Final[tuple[str, ...]] = ("joint", "marker", "both")
#: Stable per-source tokens used in the cell seed, independent of grid order.
SOURCE_TOKENS: Final[dict[str, int]] = {"none": 0, "joint": 1, "marker": 2, "both": 3}
#: Marker translation standard deviation per axis, in millimetres per level.
MARKER_TRANSLATION_MM_PER_LEVEL = 5.0
#: Method ids used in the ``method`` column, in reporting order.
METHODS: Final[tuple[str, ...]] = (
    "hand_eye_axxb",
    "camera_to_base_closed_loop_single",
    "camera_to_base_closed_loop_mean",
    "camera_to_base_axxb",
)
#: ``--quick`` smoke-run parameters.
QUICK_TRIALS = 5
QUICK_POSE_COUNTS: Final[tuple[int, ...]] = (6, 24)
#: Where ``--quick`` writes when ``--output-dir`` is not given.
QUICK_OUTPUT_DIR: Final[Path] = Path("results") / "quick"

#: Documented arm configuration, identical to ``tests/test_eyetohand_camerabase.py``.
#: Joint values in :data:`~so101_classical_control.backend.JOINT_NAMES` order, radians.
VISIBILITY_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.1,
    "elbow_flex": 0.0,
    "wrist_flex": 1.4,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

#: Bounds used to anchor the joint-noise path to MuJoCo forward kinematics.
FK_TRANSLATION_TOL = 1e-6  # metres
FK_ROTATION_TOL = 1e-6  # radians
#: Bound used to check that the desk-fixed camera and marker do not move.
STATIC_POSE_TRANSLATION_TOL = 1e-6  # metres
STATIC_POSE_ROTATION_TOL = 1e-6  # radians

TRIAL_CSV_HEADER: Final[tuple[str, ...]] = (
    "noise_source",
    "noise_level",
    "joint_sigma_deg",
    "marker_rotation_sigma_deg",
    "marker_translation_sigma_mm",
    "method",
    "num_poses",
    "cell_pose_count",
    "trial",
    "status",
    "rotation_error_deg",
    "translation_error_mm",
)

SUMMARY_CSV_HEADER: Final[tuple[str, ...]] = (
    "noise_source",
    "noise_level",
    "joint_sigma_deg",
    "marker_rotation_sigma_deg",
    "marker_translation_sigma_mm",
    "method",
    "num_poses",
    "cell_pose_count",
    "trials",
    "solver_errors",
    "rotation_error_deg_median",
    "rotation_error_deg_p95",
    "translation_error_mm_median",
    "translation_error_mm_p95",
)

_DTYPE = np.float64


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth poses and reached joint angles from :class:`SimBackend`."""

    #: Reached arm angles, ``(N, 5)`` in :data:`ARM_JOINT_NAMES` order, radians.
    joint_positions: np.ndarray
    #: ``gripperframe`` poses, ``(N, 4, 4)`` in the base frame.
    gripper_frames: np.ndarray
    #: ``wrist_cam`` poses, ``(N, 4, 4)`` in the base frame.
    wrist_cams: np.ndarray
    #: ``ee_marker`` poses, ``(N, 4, 4)`` in the base frame.
    ee_markers: np.ndarray
    #: Static ``top_cam`` pose, ``(4, 4)``, from the first configuration.
    top_cam: np.ndarray
    #: Static ``desk_marker`` pose, ``(4, 4)``, from the first configuration.
    desk_marker: np.ndarray
    #: Eye-in-hand ground truth ``inv(gripperframe_0) @ wrist_cam_0``, ``(4, 4)``.
    hand_eye_gt: np.ndarray
    #: Rigid marker offset ``inv(gripperframe_0) @ ee_marker_0``, ``(4, 4)``.
    ee_marker_offset: np.ndarray


@dataclass(frozen=True)
class _PoseView:
    """Deterministic per-pose-count quantities reused across trials.

    Everything here is a pure function of :class:`GroundTruth` and the pose
    count ``N``, so recomputing it per trial would only repeat work: the
    noiseless forward kinematics, the base-frame marker poses it implies, and
    the two ground-truth marker measurements the noise is applied to.
    """

    #: Noiseless ``gripperframe`` poses, ``(N, 4, 4)``.
    hand_poses: tuple[np.ndarray, ...]
    #: ``hand_pose @ ee_marker_offset``, ``(N, 4, 4)``.
    base_markers: tuple[np.ndarray, ...]
    #: ``inv(top_cam) @ ee_marker``, the unperturbed ``T_cam_marker``, ``(N, 4, 4)``.
    ee_markers_in_cam: tuple[np.ndarray, ...]
    #: ``inv(wrist_cam) @ desk_marker``, the unperturbed wrist measurement, ``(N, 4, 4)``.
    desk_markers_in_wrist_cam: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TrialRow:
    """One row of the raw per-trial CSV."""

    noise_source: str
    noise_level: float
    joint_sigma_deg: float
    marker_rotation_sigma_deg: float
    marker_translation_sigma_mm: float
    method: str
    num_poses: int
    cell_pose_count: int
    trial: int
    status: str
    rotation_error_deg: float
    translation_error_mm: float


@dataclass(frozen=True)
class SummaryRow:
    """One row of the aggregated summary CSV."""

    noise_source: str
    noise_level: float
    joint_sigma_deg: float
    marker_rotation_sigma_deg: float
    marker_translation_sigma_mm: float
    method: str
    num_poses: int
    cell_pose_count: int
    trials: int
    solver_errors: int
    rotation_error_deg_median: float
    rotation_error_deg_p95: float
    translation_error_mm_median: float
    translation_error_mm_p95: float


@dataclass(frozen=True)
class SweepResult:
    """Paths written by :func:`run`, plus the rows behind them."""

    trials_csv: Path
    summary_csv: Path
    plot_png: Path
    trial_rows: tuple[TrialRow, ...]
    summary_rows: tuple[SummaryRow, ...]


# ---------------------------------------------------------------------------
# Measurement model (pure numpy; no MuJoCo needed)
# ---------------------------------------------------------------------------


def perturb_joints(q: np.ndarray, sigma_rad: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``q`` plus iid ``N(0, sigma_rad)`` noise per arm joint.

    This models a noisy encoder estimate: the arm physically sits at ``q``, but
    the reported angles are perturbed. ``sigma_rad == 0`` returns a copy of ``q``
    unchanged.
    """
    angles = np.asarray(q, dtype=_DTYPE)
    if sigma_rad == 0.0:
        return angles.copy()
    return angles + rng.normal(0.0, sigma_rad, angles.shape)


def perturb_pose(
    transform: np.ndarray,
    rotation_sigma_rad: float,
    translation_sigma_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``transform @ pose(R_delta, t_delta)``, a noisy SE(3) pose.

    The rotation delta is the Rodrigues rotation of a rotation vector with iid
    ``N(0, rotation_sigma_rad)`` components; the zero vector maps to the
    identity. The translation delta has iid ``N(0, translation_sigma_m)``
    components. Both sigmas zero return a bitwise-equal copy of ``transform``.

    Right-multiplication means the noise is expressed in the measured frame, not
    in the frame of the caller.
    """
    base = np.asarray(transform, dtype=_DTYPE)
    if rotation_sigma_rad == 0.0 and translation_sigma_m == 0.0:
        return base.copy()

    if rotation_sigma_rad == 0.0:
        rotation_delta = np.eye(3, dtype=_DTYPE)
    else:
        rotation_vector = rng.normal(0.0, rotation_sigma_rad, 3)
        norm = float(np.linalg.norm(rotation_vector))
        rotation_delta = (
            np.eye(3, dtype=_DTYPE)
            if norm == 0.0
            else axis_angle_to_rotation(rotation_vector / norm, norm)
        )

    translation_delta = (
        np.zeros(3, dtype=_DTYPE)
        if translation_sigma_m == 0.0
        else rng.normal(0.0, translation_sigma_m, 3)
    )

    return base @ pose(rotation_delta, translation_delta)


def average_poses(poses: Sequence[np.ndarray]) -> np.ndarray:
    """Return the chordal L2 mean of a sequence of SE(3) poses.

    The translation is the arithmetic mean. The rotation is the projection of
    the sum of the rotation matrices onto SO(3), ``S = sum_k R_k = U Sigma V^T``,
    ``R = U V^T`` with the sign of the last column of ``U`` flipped when
    ``det(U V^T) < 0``. This aggregates the per-pose closed-loop estimates of
    ``camera_to_base_per_pose``; it is not a new calibration solver.

    Raises
    ------
    ValueError
        If ``poses`` is empty.
    """
    items = [np.asarray(item, dtype=_DTYPE) for item in poses]
    if not items:
        raise ValueError("poses must not be empty")

    translation = np.mean([item[:3, 3] for item in items], axis=0)
    rotation_sum = np.sum([item[:3, :3] for item in items], axis=0)
    u_matrix, _, vt_matrix = np.linalg.svd(rotation_sum)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0.0:
        u_matrix = u_matrix.copy()
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix
    return pose(rotation, translation)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rotation_error_deg(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return the angle in degrees of ``R_est.T @ R_gt``.

    The ``arctan2`` form ``arctan2(||vee(R - R^T)|| / 2, (trace R - 1) / 2)`` is
    used instead of ``arccos`` because ``arccos`` loses about half the digits
    near zero error, the same reasoning as ``tests/test_handeye_axxb.py``.
    """
    relative = estimated[:3, :3].T @ ground_truth[:3, :3]
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=_DTYPE,
    )
    sin_part = 0.5 * float(np.linalg.norm(skew))
    cos_part = 0.5 * (float(np.trace(relative)) - 1.0)
    return float(np.degrees(np.arctan2(sin_part, cos_part)))


def translation_error_mm(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return ``1000 * ||t_est - t_gt||`` in millimetres."""
    offset = estimated[:3, 3] - ground_truth[:3, 3]
    return float(1000.0 * np.linalg.norm(offset))


# ---------------------------------------------------------------------------
# Ground-truth collection
# ---------------------------------------------------------------------------


def _joint_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    """Return the compiled joint range of ``name``."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise RuntimeError(f"scene is missing joint {name!r}")
    low, high = model.jnt_range[joint_id]
    return float(low), float(high)


def _configurations(
    model: mujoco.MjModel, rng: np.random.Generator, count: int
) -> list[dict[str, float]]:
    """Return the documented configurations of the calibration tests.

    ``q = 0``, :data:`VISIBILITY_POSE`, the mid-range pose of the five arm
    joints, and seeded uniform samples inside every joint stop (all six joints,
    gripper included) minus :data:`LIMIT_MARGIN`.
    """
    mid_range = {name: sum(_joint_range(model, name)) / 2.0 for name in JOINT_NAMES[:5]}
    configurations: list[dict[str, float]] = [
        dict.fromkeys(JOINT_NAMES, 0.0),
        dict(VISIBILITY_POSE),
        {**mid_range, "gripper": 0.0},
    ]
    for _ in range(count - len(configurations)):
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


def collect_ground_truth(
    configuration_count: int = CONFIGURATION_COUNT,
    seed: int = CONFIGURATION_SEED,
) -> GroundTruth:
    """Collect ground-truth joint angles and poses from :class:`SimBackend`.

    For every configuration the arm is commanded, settled with ``step(1.0)``, and
    the joint positions actually reached are recorded, together with the site and
    camera poses. :func:`~so101_classical_control.kinematics.forward_kinematics`
    must reproduce the recorded ``gripperframe`` pose at
    :data:`FK_TRANSLATION_TOL` / :data:`FK_ROTATION_TOL`; otherwise the joint
    noise path would not be anchored to MuJoCo and the function raises
    ``RuntimeError``. ``top_cam`` and ``desk_marker`` must additionally be static
    across configurations within :data:`STATIC_POSE_TRANSLATION_TOL` /
    :data:`STATIC_POSE_ROTATION_TOL`.

    Raises
    ------
    RuntimeError
        If the MuJoCo ``gripperframe`` pose and the DH forward kinematics
        disagree, if the desk-fixed camera or marker moves, or if the scene is
        missing a joint.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    configurations = _configurations(model, np.random.default_rng(seed), configuration_count)

    joint_positions: list[np.ndarray] = []
    gripper_frames: list[np.ndarray] = []
    wrist_cams: list[np.ndarray] = []
    ee_markers: list[np.ndarray] = []
    top_cams: list[np.ndarray] = []
    desk_markers: list[np.ndarray] = []

    with SimBackend() as backend:
        for targets in configurations:
            backend.command_joint_targets(targets)
            backend.step(1.0)
            positions = backend.read_joint_positions()
            q = np.array([positions[name] for name in ARM_JOINT_NAMES], dtype=_DTYPE)
            gripper_frame = backend.get_site_pose("gripperframe")

            fk_pose = forward_kinematics(q)
            fk_translation = float(np.linalg.norm(fk_pose[:3, 3] - gripper_frame[:3, 3]))
            fk_rotation = _rotation_angle_rad(fk_pose, gripper_frame)
            if fk_translation > FK_TRANSLATION_TOL or fk_rotation > FK_ROTATION_TOL:
                raise RuntimeError(
                    "forward kinematics does not reproduce the MuJoCo gripperframe pose: "
                    f"{fk_translation:.3e} m, {fk_rotation:.3e} rad"
                )

            joint_positions.append(q)
            gripper_frames.append(gripper_frame)
            wrist_cams.append(backend.get_camera_pose("wrist_cam"))
            ee_markers.append(backend.get_site_pose("ee_marker"))
            top_cams.append(backend.get_camera_pose("top_cam"))
            desk_markers.append(backend.get_site_pose("desk_marker"))

    _assert_static("top_cam", top_cams)
    _assert_static("desk_marker", desk_markers)

    gripper_array = np.stack(gripper_frames)
    wrist_array = np.stack(wrist_cams)
    ee_array = np.stack(ee_markers)

    return GroundTruth(
        joint_positions=np.stack(joint_positions),
        gripper_frames=gripper_array,
        wrist_cams=wrist_array,
        ee_markers=ee_array,
        top_cam=top_cams[0],
        desk_marker=desk_markers[0],
        hand_eye_gt=inverse_se3(gripper_array[0]) @ wrist_array[0],
        ee_marker_offset=inverse_se3(gripper_array[0]) @ ee_array[0],
    )


def _assert_static(name: str, poses: Sequence[np.ndarray]) -> None:
    """Raise ``RuntimeError`` if ``poses`` varies by more than the static bounds."""
    reference = poses[0]
    for index, candidate in enumerate(poses[1:], start=1):
        translation = float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
        rotation = _rotation_angle_rad(reference, candidate)
        if translation > STATIC_POSE_TRANSLATION_TOL or rotation > STATIC_POSE_ROTATION_TOL:
            raise RuntimeError(
                f"{name} is not static across configurations: configuration {index} moves "
                f"{translation:.3e} m and {rotation:.3e} rad from configuration 0"
            )


def _rotation_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    """Return the rotation angle in radians of ``left.R.T @ right.R``."""
    relative = left[:3, :3].T @ right[:3, :3]
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=_DTYPE,
    )
    sin_part = 0.5 * float(np.linalg.norm(skew))
    cos_part = 0.5 * (float(np.trace(relative)) - 1.0)
    return float(np.arctan2(sin_part, cos_part))


def _build_pose_view(ground_truth: GroundTruth, num_poses: int) -> _PoseView:
    """Return the deterministic per-pose-count quantities for the first ``N`` poses."""
    top_cam_inverse = inverse_se3(ground_truth.top_cam)
    hand_poses: list[np.ndarray] = []
    base_markers: list[np.ndarray] = []
    ee_markers_in_cam: list[np.ndarray] = []
    desk_markers_in_wrist_cam: list[np.ndarray] = []
    for index in range(num_poses):
        hand_pose = forward_kinematics(ground_truth.joint_positions[index])
        hand_poses.append(hand_pose)
        base_markers.append(hand_pose @ ground_truth.ee_marker_offset)
        ee_markers_in_cam.append(top_cam_inverse @ ground_truth.ee_markers[index])
        desk_markers_in_wrist_cam.append(
            inverse_se3(ground_truth.wrist_cams[index]) @ ground_truth.desk_marker
        )
    return _PoseView(
        hand_poses=tuple(hand_poses),
        base_markers=tuple(base_markers),
        ee_markers_in_cam=tuple(ee_markers_in_cam),
        desk_markers_in_wrist_cam=tuple(desk_markers_in_wrist_cam),
    )


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


def _noise_sigmas(noise_source: str, noise_level: float) -> tuple[float, float, float]:
    """Return ``(joint_deg, marker_rotation_deg, marker_translation_mm)``.

    ``noise_source`` is one of ``"none"``, ``"joint"``, ``"marker"``, and
    ``"both"``; the switched-off channels stay exactly zero.
    """
    if noise_source in ("joint", "both"):
        joint_deg = noise_level
    else:
        joint_deg = 0.0
    if noise_source in ("marker", "both"):
        marker_rotation_deg = noise_level
        marker_translation_mm = MARKER_TRANSLATION_MM_PER_LEVEL * noise_level
    else:
        marker_rotation_deg = 0.0
        marker_translation_mm = 0.0
    return joint_deg, marker_rotation_deg, marker_translation_mm


def _solver_estimate(
    a_list: Sequence[np.ndarray], b_list: Sequence[np.ndarray]
) -> tuple[np.ndarray | None, str]:
    """Return ``(estimate, status)``; a solver ``ValueError`` becomes a status."""
    try:
        return solve_ax_xb(a_list, b_list), "ok"
    except ValueError:
        return None, "solver_error"


def _compare(estimated: np.ndarray | None, ground_truth: np.ndarray) -> tuple[float, float]:
    """Return ``(rotation_error_deg, translation_error_mm)``, or NaNs if absent."""
    if estimated is None:
        return math.nan, math.nan
    return rotation_error_deg(estimated, ground_truth), translation_error_mm(
        estimated, ground_truth
    )


def _reference(method: str, ground_truth: GroundTruth) -> np.ndarray:
    """Return the ground-truth transform ``method`` is compared against."""
    if method == "hand_eye_axxb":
        return ground_truth.hand_eye_gt
    return ground_truth.top_cam


def evaluate_trial(
    ground_truth: GroundTruth,
    noise_source: str,
    noise_level: float,
    num_poses: int,
    trial: int,
    rng: np.random.Generator,
    cell_pose_count: int | None = None,
    view: _PoseView | None = None,
) -> list[TrialRow]:
    """Run all four methods once for one noise cell and one trial.

    All methods of the trial draw from ``rng``, the generator of the
    ``(noise source, level, pose count, trial)`` cell. The fixed-camera marker
    measurement is shared by the three camera-to-base methods; the wrist-camera
    marker measurement is used only by ``hand_eye_axxb``.

    ``cell_pose_count`` is the grid pose count ``N`` of the cell (default
    ``num_poses``). It labels the row, and the single-configuration method picks
    configuration ``trial % cell_pose_count`` within the cell's first ``N``
    configurations.

    ``view`` optionally supplies the deterministic per-pose-count quantities of
    :func:`_build_pose_view`; without it they are built here.
    """
    if cell_pose_count is None:
        cell_pose_count = num_poses
    if view is None:
        view = _build_pose_view(ground_truth, num_poses)

    joint_deg, marker_rotation_deg, marker_translation_mm = _noise_sigmas(noise_source, noise_level)
    joint_sigma_rad = float(np.radians(joint_deg))
    marker_rotation_sigma_rad = float(np.radians(marker_rotation_deg))
    marker_translation_sigma_m = marker_translation_mm * 1e-3

    hand_poses: list[np.ndarray] = []
    base_markers: list[np.ndarray] = []
    cam_markers: list[np.ndarray] = []
    wrist_cam_estimates: list[np.ndarray] = []

    for index in range(num_poses):
        if joint_sigma_rad == 0.0:
            hand_pose = view.hand_poses[index]
            base_marker = view.base_markers[index]
        else:
            q_noisy = perturb_joints(ground_truth.joint_positions[index], joint_sigma_rad, rng)
            hand_pose = forward_kinematics(q_noisy)
            base_marker = hand_pose @ ground_truth.ee_marker_offset
        hand_poses.append(hand_pose)
        base_markers.append(base_marker)

        cam_markers.append(
            perturb_pose(
                view.ee_markers_in_cam[index],
                marker_rotation_sigma_rad,
                marker_translation_sigma_m,
                rng,
            )
        )

        wrist_cam_marker = perturb_pose(
            view.desk_markers_in_wrist_cam[index],
            marker_rotation_sigma_rad,
            marker_translation_sigma_m,
            rng,
        )
        wrist_cam_estimates.append(ground_truth.desk_marker @ inverse_se3(wrist_cam_marker))

    rows: list[TrialRow] = []

    def add_row(method: str, num: int, estimated: np.ndarray | None, status: str) -> None:
        if status == "ok" and estimated is not None:
            rotation_deg, translation_mm = _compare(estimated, _reference(method, ground_truth))
        else:
            rotation_deg, translation_mm = math.nan, math.nan
        rows.append(
            TrialRow(
                noise_source=noise_source,
                noise_level=noise_level,
                joint_sigma_deg=joint_deg,
                marker_rotation_sigma_deg=marker_rotation_deg,
                marker_translation_sigma_mm=marker_translation_mm,
                method=method,
                num_poses=num,
                cell_pose_count=cell_pose_count,
                trial=trial,
                status=status,
                rotation_error_deg=rotation_deg,
                translation_error_mm=translation_mm,
            )
        )

    a_list, b_list = build_motion_pairs(hand_poses, wrist_cam_estimates)
    hand_eye_estimate, hand_eye_status = _solver_estimate(a_list, b_list)
    add_row("hand_eye_axxb", num_poses, hand_eye_estimate, hand_eye_status)

    index = trial % cell_pose_count
    single_estimate = camera_to_base(base_markers[index], cam_markers[index])
    add_row("camera_to_base_closed_loop_single", 1, single_estimate, "ok")

    mean_estimate = average_poses(camera_to_base_per_pose(base_markers, cam_markers))
    add_row("camera_to_base_closed_loop_mean", num_poses, mean_estimate, "ok")

    a_list, b_list = build_fixed_frame_motion_pairs(base_markers, cam_markers)
    ax_xb_estimate, ax_xb_status = _solver_estimate(a_list, b_list)
    add_row("camera_to_base_axxb", num_poses, ax_xb_estimate, ax_xb_status)

    return rows


# ---------------------------------------------------------------------------
# Aggregation and CSV output
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """Format a float for the CSVs as ``%.6g``, with ``nan`` for non-finite."""
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _percentile(values: list[float], percentile: float) -> float:
    """Return a percentile of ``values`` or NaN when there are none.

    ``numpy.percentile`` uses linear interpolation between the two nearest
    order statistics (its default ``method="linear"``), which is what the
    documented p95 column is.
    """
    if not values:
        return math.nan
    return float(np.percentile(values, percentile))


def _summarize(trial_rows: Sequence[TrialRow]) -> list[SummaryRow]:
    """Aggregate raw rows into one summary row per cell and method.

    Groups are keyed by ``(noise_source, noise_level, method, num_poses,
    cell_pose_count)``; ``cell_pose_count`` keeps
    ``camera_to_base_closed_loop_single`` rows from different pose-count cells
    separate even though they all report ``num_poses = 1``.
    """
    groups: dict[tuple[str, float, str, int, int], list[TrialRow]] = {}
    for row in trial_rows:
        key = (
            row.noise_source,
            row.noise_level,
            row.method,
            row.num_poses,
            row.cell_pose_count,
        )
        groups.setdefault(key, []).append(row)

    source_rank = {"none": 0, "joint": 1, "marker": 2, "both": 3}
    method_rank = {method: index for index, method in enumerate(METHODS)}
    ordered_keys = sorted(
        groups,
        key=lambda key: (
            source_rank.get(key[0], 99),
            key[1],
            key[4],
            method_rank.get(key[2], 99),
        ),
    )

    summary: list[SummaryRow] = []
    for key in ordered_keys:
        group = groups[key]
        usable = [row for row in group if row.status == "ok"]
        rotation_values = [row.rotation_error_deg for row in usable]
        translation_values = [row.translation_error_mm for row in usable]
        first = group[0]
        summary.append(
            SummaryRow(
                noise_source=key[0],
                noise_level=key[1],
                joint_sigma_deg=first.joint_sigma_deg,
                marker_rotation_sigma_deg=first.marker_rotation_sigma_deg,
                marker_translation_sigma_mm=first.marker_translation_sigma_mm,
                method=key[2],
                num_poses=key[3],
                cell_pose_count=key[4],
                trials=len(group),
                solver_errors=len(group) - len(usable),
                rotation_error_deg_median=_percentile(rotation_values, 50.0),
                rotation_error_deg_p95=_percentile(rotation_values, 95.0),
                translation_error_mm_median=_percentile(translation_values, 50.0),
                translation_error_mm_p95=_percentile(translation_values, 95.0),
            )
        )
    return summary


def _write_trial_csv(path: Path, rows: Sequence[TrialRow]) -> None:
    """Write the raw per-trial CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TRIAL_CSV_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.noise_source,
                    _format_float(row.noise_level),
                    _format_float(row.joint_sigma_deg),
                    _format_float(row.marker_rotation_sigma_deg),
                    _format_float(row.marker_translation_sigma_mm),
                    row.method,
                    str(row.num_poses),
                    str(row.cell_pose_count),
                    str(row.trial),
                    row.status,
                    _format_float(row.rotation_error_deg),
                    _format_float(row.translation_error_mm),
                ]
            )


def _write_summary_csv(path: Path, rows: Sequence[SummaryRow]) -> None:
    """Write the committed summary CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_CSV_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.noise_source,
                    _format_float(row.noise_level),
                    _format_float(row.joint_sigma_deg),
                    _format_float(row.marker_rotation_sigma_deg),
                    _format_float(row.marker_translation_sigma_mm),
                    row.method,
                    str(row.num_poses),
                    str(row.cell_pose_count),
                    str(row.trials),
                    str(row.solver_errors),
                    _format_float(row.rotation_error_deg_median),
                    _format_float(row.rotation_error_deg_p95),
                    _format_float(row.translation_error_mm_median),
                    _format_float(row.translation_error_mm_p95),
                ]
            )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _select_row(
    summary_rows: Sequence[SummaryRow],
    source: str,
    method: str,
    cell_pose_count: int,
    level: float,
) -> SummaryRow | None:
    """Return the summary row for one (source, method, cell pose count, level)."""
    for row in summary_rows:
        if (
            row.noise_source == source
            and row.method == method
            and row.noise_level == level
            and row.cell_pose_count == cell_pose_count
        ):
            return row
    return None


def _plot_series(
    summary_rows: Sequence[SummaryRow],
    source: str,
    method: str,
    cell_pose_count: int,
    field: str,
    positive_levels: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Return ``(levels, errors)`` for one series; drop non-positive errors."""
    level_values: list[float] = []
    error_values: list[float] = []
    for level in positive_levels:
        row = _select_row(summary_rows, source, method, cell_pose_count, level)
        if row is None:
            continue
        value = float(getattr(row, field))
        if not math.isfinite(value) or value <= 0.0:
            continue
        level_values.append(level)
        error_values.append(value)
    return level_values, error_values


def _plot_summary(
    summary_rows: Sequence[SummaryRow],
    path: Path,
    pose_counts: Sequence[int],
    noise_levels: Sequence[float],
) -> None:
    """Save the 2x3 summary plot, importing matplotlib only when called.

    Rows are the median rotation and translation errors, columns the noise
    sources ``joint``, ``marker``, and ``both``. Solid lines use the largest pose
    count, and ``hand_eye_axxb`` at the smallest pose count is dashed. The error
    axis is logarithmic, so level ``0`` (errors around ``1e-12``) and any
    non-finite values are omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = ("joint", "marker", "both")
    metrics = (
        ("rotation_error_deg_median", "median rotation error (deg)"),
        ("translation_error_mm_median", "median translation error (mm)"),
    )
    solid_count = max(pose_counts)
    dashed_count = min(pose_counts)
    positive_levels = [level for level in noise_levels if level > 0.0]

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for column, source in enumerate(sources):
        for row_index, (field, metric_label) in enumerate(metrics):
            axis = axes[row_index][column]
            for method in METHODS:
                level_values, error_values = _plot_series(
                    summary_rows,
                    source,
                    method,
                    solid_count,
                    field,
                    positive_levels,
                )
                if level_values:
                    series_label = (
                        method
                        if method == "camera_to_base_closed_loop_single"
                        else f"{method} (N={solid_count})"
                    )
                    axis.plot(level_values, error_values, marker="o", label=series_label)
            if dashed_count != solid_count:
                level_values, error_values = _plot_series(
                    summary_rows,
                    source,
                    "hand_eye_axxb",
                    dashed_count,
                    field,
                    positive_levels,
                )
                if level_values:
                    axis.plot(
                        level_values,
                        error_values,
                        marker="s",
                        linestyle="--",
                        label=f"hand_eye_axxb (N={dashed_count})",
                    )
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.3)
            axis.set_title(f"noise source: {source}")
            if column == 0:
                axis.set_ylabel(metric_label)
            if row_index == 1:
                axis.set_xlabel(
                    "noise level s (joint: s deg, marker: s deg, "
                    f"{MARKER_TRANSLATION_MM_PER_LEVEL:g}s mm)"
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7)
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _validate_parameters(
    trials: int, seed: int, pose_counts: Sequence[int], noise_levels: Sequence[float]
) -> None:
    """Raise ``ValueError`` for an invalid sweep parameter set."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if not pose_counts:
        raise ValueError("pose_counts must not be empty")
    if min(pose_counts) < 2:
        raise ValueError("each pose count must be at least two")
    if max(pose_counts) > CONFIGURATION_COUNT:
        raise ValueError(
            f"pose counts cannot exceed the {CONFIGURATION_COUNT} ground-truth configurations"
        )
    if not noise_levels:
        raise ValueError("noise_levels must not be empty")


def run(
    output_dir: str | Path = "results",
    trials: int = TRIALS,
    seed: int = SEED,
    pose_counts: Sequence[int] = POSE_COUNTS,
    noise_levels: Sequence[float] = NOISE_LEVELS,
    ground_truth: GroundTruth | None = None,
) -> SweepResult:
    """Run the sweep, write the two CSVs and the PNG, and print a summary.

    One generator is created per ``(noise source, level, pose count, trial)``
    cell from stable seed tokens, so every cell is reproducible independently of
    execution order and of which other cells are run. Level zero is run only
    once (as ``noise_source="none"``) instead of once per source, because the
    three sources coincide there.

    ``ground_truth`` optionally supplies already-collected poses (for example
    from a session-scoped test fixture) so :func:`collect_ground_truth` is not
    repeated. The deterministic per-pose-count quantities are computed once per
    distinct pose count and reused by every trial.

    Returns the paths and rows behind the written artifacts.
    """
    _validate_parameters(trials, seed, pose_counts, noise_levels)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if ground_truth is None:
        ground_truth = collect_ground_truth(CONFIGURATION_COUNT, CONFIGURATION_SEED)

    views = {num_poses: _build_pose_view(ground_truth, num_poses) for num_poses in pose_counts}

    trial_rows: list[TrialRow] = []
    for source in NOISE_SOURCES:
        for level in noise_levels:
            if level == 0.0 and source != NOISE_SOURCES[0]:
                continue
            source_label = "none" if level == 0.0 else source
            for num_poses in pose_counts:
                for trial in range(trials):
                    rng = np.random.default_rng(
                        [
                            seed,
                            SOURCE_TOKENS[source_label],
                            round(level * 1000),
                            num_poses,
                            trial,
                        ]
                    )
                    trial_rows.extend(
                        evaluate_trial(
                            ground_truth,
                            source_label,
                            level,
                            num_poses,
                            trial,
                            rng,
                            cell_pose_count=num_poses,
                            view=views[num_poses],
                        )
                    )

    summary_rows = _summarize(trial_rows)
    trials_csv = output / "noise_robustness_trials.csv"
    summary_csv = output / "noise_robustness_summary.csv"
    plot_png = output / "noise_robustness.png"
    _write_trial_csv(trials_csv, trial_rows)
    _write_summary_csv(summary_csv, summary_rows)
    _plot_summary(summary_rows, plot_png, pose_counts, noise_levels)
    _print_summary(summary_rows, pose_counts, noise_levels)

    return SweepResult(
        trials_csv=trials_csv,
        summary_csv=summary_csv,
        plot_png=plot_png,
        trial_rows=tuple(trial_rows),
        summary_rows=tuple(summary_rows),
    )


def _print_summary(
    summary_rows: Sequence[SummaryRow],
    pose_counts: Sequence[int],
    noise_levels: Sequence[float],
) -> None:
    """Print the median errors at level 1.0, largest pose count, source both."""
    report_level = 1.0
    report_poses = max(pose_counts)
    if report_level not in noise_levels:
        return
    print(
        f"noise robustness: source=both, level={report_level:g}, "
        f"N={report_poses}, per method (median):"
    )
    for method in METHODS:
        row = _select_row(summary_rows, "both", method, report_poses, report_level)
        if row is None:
            continue
        print(
            f"  {row.method}: rotation {row.rotation_error_deg_median:.3f} deg, "
            f"translation {row.translation_error_mm_median:.3f} mm, "
            f"solver errors {row.solver_errors}/{row.trials}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_output_dir(output_dir: str | Path | None, quick: bool) -> str:
    """Return the output directory for a CLI invocation.

    An explicit ``--output-dir`` always wins. Otherwise ``--quick`` writes to
    :data:`QUICK_OUTPUT_DIR` (``results/quick``) so a smoke run never overwrites
    the committed artifacts, and a normal run writes to ``results``.
    """
    if output_dir is not None:
        return str(output_dir)
    return str(QUICK_OUTPUT_DIR) if quick else "results"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep from the command line and return an exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Noise-robustness sweep for the eye-in-hand and camera-to-base "
            "calibrations (Issue #10)."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "directory for the two CSVs and the PNG "
            "(default: results, or results/quick with --quick)"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TRIALS,
        help=f"trials per grid cell (default: {TRIALS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed (default: {SEED})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"short smoke run: {QUICK_TRIALS} trials, pose counts {QUICK_POSE_COUNTS}",
    )
    arguments = parser.parse_args(argv)

    output_dir = resolve_output_dir(arguments.output_dir, arguments.quick)
    if arguments.quick:
        run(
            output_dir,
            trials=QUICK_TRIALS,
            seed=arguments.seed,
            pose_counts=QUICK_POSE_COUNTS,
        )
    else:
        run(output_dir, trials=arguments.trials, seed=arguments.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
