"""L03 goal-function demonstration scenarios in the MuJoCo sponge-to-paper scene.

This module drives the goal-level functions in
:mod:`so101_classical_control.goals` on the shared scene and records per-frame
CSV artifacts. It is a demonstration and logging harness, not a controller and
not a new solver: every motion command comes from a goal function, and the
simulation runs through the public :class:`~so101_classical_control.sim.SimBackend`
API (``command_joint_targets`` plus ``step``).

Run it as::

    python -m so101_classical_control.sim.l03_demo \
        --scenario {fk_sweep,ik_reach,singularity,damped_ik,camera_to_base,pick_and_place,all} \
        --out results/l03_sim/

How-to lives in ``--help``, not in a document. Rendering is off by default and
the scenarios are headless; ``--render`` is an optional hook that writes each
scenario's recorded ``top_cam`` frames to an mp4 and needs a working OpenGL
backend.

Per-frame CSV
-------------
Every scenario writes ``<scenario>.csv`` with this header::

    frame,time_s,shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,
    gripper,our_px,our_py,our_pz,our_qw,our_qx,our_qy,our_qz,
    mj_px,mj_py,mj_pz,mj_qw,mj_qx,mj_qy,mj_qz,manipulability,note

The joint columns are the positions **actually read back** after settling, not
the commanded targets. ``our_*`` is
:func:`~so101_classical_control.goals.locate_gripper_from_joint_angles` of those
positions; ``mj_*`` is ``get_site_pose("gripperframe")`` at the same instant. The
position columns are metres in the robot base frame. Quaternions are in
``(w, x, y, z)`` order, the order the repository already uses, and are the frame
rotation of the pose (not the pose's inverse). ``manipulability`` is
:func:`~so101_classical_control.goals.report_manipulability`; ``note`` is free
text such as ``target_ok``, ``unreachable_hold``, or ``hold``.

``pick_and_place`` writes the same CSV plus
``pick_and_place_outcome.txt`` with a ``key=value`` line per fact: the success
flag, the completion simulation time, and the final sponge and paper positions.
No number there is an estimate; each is read from the simulation. The task is a
skeleton: whether the free-joint sponge is actually captured by the jaws is
reported honestly, and a failed attempt is not recorded as a success.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from ..backend import JOINT_NAMES
from ..goals import (
    UnreachableTarget,
    carry_to_paper,
    close_gripper_on_object,
    keep_gripper_fixed_to_first_order,
    locate_gripper_from_joint_angles,
    locate_sponge_in_base_frame,
    locate_top_camera_in_base_frame,
    map_joint_velocity_to_gripper_twist,
    plan_grasp_pose,
    reach_target_pose,
    release_object,
    report_manipulability,
)
from ..kinematics import (
    ARM_JOINT_NAMES,
    forward_kinematics,
    inverse_se3,
    rotation_to_quaternion,
)
from ..sim import SimBackend
from ..sim.scene import scene_xml_path

__all__ = [
    "CSV_HEADER",
    "PICK_AND_PLACE_OUTCOME",
    "SCENARIOS",
    "SettleConfig",
    "main",
    "run_scenario",
]

#: Per-frame CSV header, documented in the module docstring.
CSV_HEADER: Final[tuple[str, ...]] = (
    "frame",
    "time_s",
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "our_px",
    "our_py",
    "our_pz",
    "our_qw",
    "our_qx",
    "our_qy",
    "our_qz",
    "mj_px",
    "mj_py",
    "mj_pz",
    "mj_qw",
    "mj_qx",
    "mj_qy",
    "mj_qz",
    "manipulability",
    "note",
)

#: Outcome file written by the ``pick_and_place`` scenario.
PICK_AND_PLACE_OUTCOME: Final[str] = "pick_and_place_outcome.txt"

#: Names accepted by ``--scenario``. ``all`` runs all of them.
SCENARIOS: Final[tuple[str, ...]] = (
    "fk_sweep",
    "ik_reach",
    "singularity",
    "damped_ik",
    "camera_to_base",
    "pick_and_place",
)

#: Simulated seconds per control step.
FRAME_DT: Final[float] = 0.02
#: Simulated seconds to settle a position actuator before recording a frame.
SETTLE_S: Final[float] = 1.0
#: Frames per second of a rendered mp4; one frame per logged CSV row.
VIDEO_FPS: Final[int] = 20
#: Inset of the commanded ``fk_sweep`` endpoints from ``ctrlrange``, radians.
#: MuJoCo soft joint limits can let a position-controlled joint settle slightly
#: past ``jnt_range`` when the actuator is commanded to the exact ``ctrlrange``
#: endpoint (observed overshoot 6.13e-6 to 2.60e-4 rad on the Issue #30 run).
#: 1e-3 rad is larger than that overshoot and larger than the ctrlrange vs
#: jnt_range float mismatch, so settled ``qpos`` stays inside ``jnt_range``
#: without loosening the limit check.
FK_SWEEP_ENDPOINT_INSET: Final[float] = 1e-3
#: Half-extents of the scene's sponge and paper, metres, used by the task test.
SPONGE_HALF_EXTENTS: Final[tuple[float, float]] = (0.045, 0.03)
PAPER_HALF_EXTENTS: Final[tuple[float, float]] = (0.06, 0.045)
#: Highest sponge centre considered "resting near the desk", metres.
REST_HEIGHT: Final[float] = 0.05


@dataclass(frozen=True)
class SettleConfig:
    """How the demo commands and advances the simulation.

    Attributes
    ----------
    frame_dt
        Simulated seconds per control step.
    settle_s
        Simulated time to let a position actuator reach its target before a row
        is treated as settled.
    """

    frame_dt: float = FRAME_DT
    settle_s: float = SETTLE_S


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _format(value: float) -> str:
    """Format a float for the CSV as ``%.9g``, keeping non-finite as ``nan``."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.9g}"


def _quaternion_entry(transform: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(qw, qx, qy, qz)`` of a pose's rotation block."""
    quaternion = rotation_to_quaternion(transform[:3, :3])
    return (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))


def _read_angles(positions: dict[str, float]) -> np.ndarray:
    """Return the five arm angles of a joint mapping."""
    return np.array([positions[name] for name in ARM_JOINT_NAMES], dtype=np.float64)


def _arm_targets(angles: Sequence[float], gripper: float = 0.0) -> dict[str, float]:
    """Return a full joint mapping from five arm angles plus a gripper value."""
    return {
        **{name: float(value) for name, value in zip(ARM_JOINT_NAMES, angles, strict=True)},
        "gripper": float(gripper),
    }


def _ctrl_limits() -> dict[str, tuple[float, float]]:
    """Return the actuator ``ctrlrange`` of every joint, read from the scene."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    ranges: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        low, high = model.actuator_ctrlrange[actuator_id]
        ranges[name] = (float(low), float(high))
    return ranges


def _arm_limits() -> tuple[tuple[float, float], ...]:
    """Return the five arm-joint limit pairs, used by the IK calls."""
    limits = _ctrl_limits()
    return tuple(limits[name] for name in ARM_JOINT_NAMES)


def _clamp_arm(angles: Sequence[float], limits: Sequence[tuple[float, float]]) -> np.ndarray:
    """Clamp arm angles into ``limits`` for use as a *command*, not as an IK result."""
    return np.array(
        [
            min(max(float(value), low), high)
            for value, (low, high) in zip(angles, limits, strict=True)
        ],
        dtype=np.float64,
    )


def _interpolate(start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
    """Return ``steps`` linear intermediate configurations from ``start`` to ``end``."""
    return [start + (end - start) * (index / steps) for index in range(1, steps + 1)]


class _Recorder:
    """Write one CSV row per recorded frame."""

    def __init__(self, path: Path, frames: list[np.ndarray] | None = None) -> None:
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        self._writer.writerow(CSV_HEADER)
        self._frame = 0
        self.rows = 0
        self._frames = frames

    def record(self, backend: SimBackend, positions: dict[str, float], note: str) -> None:
        """Record the current settled state of ``backend`` with ``note``."""
        angles = _read_angles(positions)
        ours = locate_gripper_from_joint_angles(angles)
        mujoco_pose = backend.get_site_pose("gripperframe")
        our_q = _quaternion_entry(ours)
        mj_q = _quaternion_entry(mujoco_pose)
        manipulability = report_manipulability(angles)
        self._writer.writerow(
            [
                str(self._frame),
                _format(backend.time_s),
                *(_format(float(positions[name])) for name in ARM_JOINT_NAMES),
                _format(float(positions["gripper"])),
                *(_format(float(value)) for value in ours[:3, 3]),
                *(_format(value) for value in our_q),
                *(_format(float(value)) for value in mujoco_pose[:3, 3]),
                *(_format(value) for value in mj_q),
                _format(manipulability),
                note,
            ]
        )
        self._frame += 1
        self.rows += 1
        if self._frames is not None:
            self._frames.append(backend.get_camera_image("top_cam"))

    def close(self) -> None:
        """Flush and close the CSV."""
        self._handle.close()

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _drive(
    backend: SimBackend, targets: dict[str, float], config: SettleConfig
) -> dict[str, float]:
    """Command ``targets``, settle, and return the positions actually read."""
    backend.command_joint_targets(targets)
    backend.step(config.settle_s)
    return backend.read_joint_positions()


def _workspace_configurations(limits: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    """Return a few in-limit arm configurations used as FK targets."""
    mid = np.array([0.5 * (low + high) for low, high in limits])
    return [
        np.zeros(len(ARM_JOINT_NAMES)),
        mid,
        np.clip(
            np.array([0.0, -0.6, 0.6, 1.2, 0.0]),
            [low for low, _ in limits],
            [high for _, high in limits],
        ),
        np.clip(
            np.array([0.4, 0.2, -0.4, 1.4, -0.5]),
            [low for low, _ in limits],
            [high for _, high in limits],
        ),
    ]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _scenario_fk_sweep(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Sweep the arm joints one at a time and record the gripper workspace."""
    ctrl = _ctrl_limits()
    base_angles = _read_angles(backend.read_joint_positions())

    with _Recorder(out / "fk_sweep.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")
        for joint_index, name in enumerate(ARM_JOINT_NAMES):
            low, high = ctrl[name]
            # Command an inset strictly inside the range: the exact endpoints
            # settle past ``jnt_range`` against MuJoCo's soft joint limits.
            low += FK_SWEEP_ENDPOINT_INSET
            high -= FK_SWEEP_ENDPOINT_INSET
            for value in np.linspace(low, high, 7):
                target = base_angles.copy()
                target[joint_index] = float(min(max(value, low), high))
                positions = _drive(backend, _arm_targets(target), config)
                recorder.record(backend, positions, f"sweep:{name}")
    return {"csv": "fk_sweep.csv"}


def _scenario_ik_reach(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Reach targets obtained from FK, then refuse one far target without moving."""
    limits = _arm_limits()
    reached = 0
    refused_hold = False

    with _Recorder(out / "ik_reach.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")
        for index, angles in enumerate(_workspace_configurations(limits)):
            seed = _read_angles(backend.read_joint_positions())
            solution = reach_target_pose(forward_kinematics(angles), q0=seed, joint_limits=limits)
            if isinstance(solution, UnreachableTarget):
                recorder.record(backend, backend.read_joint_positions(), f"unreachable:{index}")
                continue
            positions = _drive(backend, _arm_targets(_clamp_arm(solution, limits)), config)
            recorder.record(backend, positions, f"target_ok:{index}")
            reached += 1

        # A far target must be refused, and the arm must stay at the
        # configuration it had before the refusal (Issue #30, clip b).
        before = backend.read_joint_positions()
        refusal = reach_target_pose([1.0, 0.0, 0.0], q0=_read_angles(before), joint_limits=limits)
        if isinstance(refusal, UnreachableTarget):
            backend.step(config.settle_s)
            after = backend.read_joint_positions()
            moved = max(abs(after[name] - before[name]) for name in ARM_JOINT_NAMES)
            refused_hold = moved <= 1e-6
            note = "unreachable_hold" if refused_hold else "unreachable_moved"
        else:
            note = "unexpected_solution"
        recorder.record(backend, backend.read_joint_positions(), note)
    return {"reached": str(reached), "refusal_held": str(refused_hold).lower()}


def _scenario_singularity(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Pass near a low-manipulability SO-101 configuration and record the measure."""
    limits = _arm_limits()
    start_angles = _read_angles(backend.read_joint_positions())
    # A near-singular pass found by scanning the thin-Jacobian measure: the
    # elbow folds back toward the base, so the arm is close to losing a task
    # direction. The value is a model configuration, not an approved threshold.
    near_singular = _clamp_arm(np.array([0.0, -0.5, -1.06, 0.0, 0.0]), limits)
    minimum = np.inf

    with _Recorder(out / "singularity.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")
        for step_angles in _interpolate(start_angles, near_singular, 6):
            positions = _drive(backend, _arm_targets(step_angles), config)
            recorder.record(backend, positions, "approach")
            minimum = min(minimum, report_manipulability(_read_angles(positions)))
        for offset in (0.0, 0.05, -0.05, 0.1, -0.1):
            angled = near_singular.copy()
            angled[1] += offset
            positions = _drive(backend, _arm_targets(_clamp_arm(angled, limits)), config)
            recorder.record(backend, positions, "near_singular")
            minimum = min(minimum, report_manipulability(_read_angles(positions)))
    return {"min_manipulability": _format(minimum)}


def _scenario_damped_ik(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Drive damped pseudo-inverse IK toward a reachable target, logging the null-space part."""
    limits = _arm_limits()
    target = forward_kinematics(np.array([0.3, -0.2, 0.4, 1.2, 0.0]))
    current = _read_angles(backend.read_joint_positions())
    steps = 0
    final_error = float("nan")

    with _Recorder(out / "damped_ik.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")
        for _ in range(12):
            solution = reach_target_pose(target, q0=current, joint_limits=limits)
            if isinstance(solution, UnreachableTarget):
                recorder.record(backend, backend.read_joint_positions(), "unreachable")
                break
            desired = current + 0.2 * (solution - current)
            null_part = keep_gripper_fixed_to_first_order(
                current, np.ones(len(ARM_JOINT_NAMES)), damping=1e-3
            )
            twist_norm = float(
                np.linalg.norm(map_joint_velocity_to_gripper_twist(current, null_part))
            )
            positions = _drive(backend, _arm_targets(_clamp_arm(desired, limits)), config)
            note = "damped" if twist_norm <= 1e-6 else "damped_with_twist"
            recorder.record(backend, positions, note)
            current = _read_angles(positions)
            steps += 1
            final_error = float(np.linalg.norm(forward_kinematics(current)[:3, 3] - target[:3, 3]))
    return {"steps": str(steps), "final_position_error_m": _format(final_error)}


def _scenario_camera_to_base(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Map a known desk point from the top-camera frame into the base frame and compare."""
    # The desk marker is a static site with identity orientation, so its pose in
    # the base frame is known without an image. This scenario checks the frame
    # algebra only; it does no image processing.
    camera_pose = backend.get_camera_pose("top_cam")
    ee_marker = backend.get_site_pose("ee_marker")
    desk_marker = backend.get_site_pose("desk_marker")

    camera_inverse = inverse_se3(camera_pose)
    # The base-frame desk point expressed in the camera frame, and the same
    # marker as seen by the camera.
    point_in_camera = camera_inverse[:3, :3] @ desk_marker[:3, 3] + camera_inverse[:3, 3]
    marker_in_camera = camera_inverse @ ee_marker

    estimated = locate_top_camera_in_base_frame(ee_marker, marker_in_camera)
    located = locate_sponge_in_base_frame(point_in_camera, T_base_cam=estimated)
    residual = float(np.linalg.norm(located - desk_marker[:3, 3]))

    with _Recorder(out / "camera_to_base.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")
        recorder.record(
            backend, backend.read_joint_positions(), f"point_residual_m={_format(residual)}"
        )
        for _ in range(3):
            backend.step(config.frame_dt)
            recorder.record(backend, backend.read_joint_positions(), "hold")
    return {"camera_to_base_residual_m": _format(residual)}


def _sponge_on_paper(sponge_pose: np.ndarray, paper_pose: np.ndarray) -> bool:
    """Return whether the sponge lies inside the paper rectangle and rests on it.

    The test is geometric, in the robot base frame: the sponge centre is inside
    the paper rectangle and the sponge is near desk height. The half-extents are
    model facts of this scene, not an approved P13 threshold.
    """
    inside_x = abs(sponge_pose[0, 3] - paper_pose[0, 3]) <= PAPER_HALF_EXTENTS[0]
    inside_y = abs(sponge_pose[1, 3] - paper_pose[1, 3]) <= PAPER_HALF_EXTENTS[1]
    near_desk = sponge_pose[2, 3] <= REST_HEIGHT
    return bool(inside_x and inside_y and near_desk)


def _write_outcome(
    path: Path,
    *,
    success: bool,
    completion_time_s: float,
    sponge_end: np.ndarray,
    paper_end: np.ndarray,
    notes: Sequence[str],
) -> None:
    """Write the ``key=value`` task-outcome file, robot base frame, metres."""
    distance = float(np.linalg.norm(sponge_end[:3, 3] - paper_end[:3, 3]))
    lines = [
        "# pick_and_place outcome; frame = robot base (baseframe), metres",
        f"success={'true' if success else 'false'}",
        f"completion_time_s={_format(completion_time_s)}",
        f"sponge_final_x={_format(float(sponge_end[0, 3]))}",
        f"sponge_final_y={_format(float(sponge_end[1, 3]))}",
        f"sponge_final_z={_format(float(sponge_end[2, 3]))}",
        f"paper_x={_format(float(paper_end[0, 3]))}",
        f"paper_y={_format(float(paper_end[1, 3]))}",
        f"paper_z={_format(float(paper_end[2, 3]))}",
        f"sponge_to_paper_m={_format(distance)}",
        f"notes={'|'.join(notes) if notes else 'none'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scenario_pick_and_place(
    backend: SimBackend,
    out: Path,
    config: SettleConfig,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Attempt the sponge-to-paper pick and place and report the outcome honestly.

    The motion is: open the gripper, move to the top-down grasp pose over the
    sponge, close, carry to the top-down place pose at the paper, and release.
    Every arm target comes from :func:`reach_target_pose` /
    :func:`carry_to_paper`; the grasp and place poses come from
    :func:`plan_grasp_pose`. If a solve reports ``UnreachableTarget`` the task
    records that and does not command a new pose. Success is not assumed: it is
    the geometric test of :func:`_sponge_on_paper` on the simulated sponge.
    """
    limits = _arm_limits()
    sponge_start = backend.get_body_pose("sponge")
    paper_start = backend.get_body_pose("paper")
    start_time = backend.time_s
    notes: list[str] = []

    with _Recorder(out / "pick_and_place.csv", frames) as recorder:
        recorder.record(backend, backend.read_joint_positions(), "start")

        backend.command_joint_targets({"gripper": release_object()})
        backend.step(config.settle_s)
        recorder.record(backend, backend.read_joint_positions(), "open")

        plan = plan_grasp_pose(sponge_start)
        grasp = reach_target_pose(
            plan.pose, q0=_read_angles(backend.read_joint_positions()), joint_limits=limits
        )
        if isinstance(grasp, UnreachableTarget):
            notes.append(f"grasp_refused:{grasp.reason}")
            recorder.record(backend, backend.read_joint_positions(), "grasp_refused")
        else:
            positions = _drive(
                backend, _arm_targets(_clamp_arm(grasp, limits)), config
            )
            recorder.record(backend, positions, "grasp")

            backend.command_joint_targets({"gripper": close_gripper_on_object()})
            backend.step(config.settle_s)
            recorder.record(backend, backend.read_joint_positions(), "close")

            place = carry_to_paper(
                paper_start,
                sponge_pose_in_base=backend.get_body_pose("sponge"),
                q0=_read_angles(backend.read_joint_positions()),
                joint_limits=limits,
            )
            if isinstance(place, UnreachableTarget):
                notes.append(f"carry_refused:{place.reason}")
                recorder.record(backend, backend.read_joint_positions(), "carry_refused")
            else:
                positions = _drive(backend, _arm_targets(_clamp_arm(place, limits)), config)
                recorder.record(backend, positions, "carry")

                backend.command_joint_targets({"gripper": release_object()})
                backend.step(config.settle_s)
                recorder.record(backend, backend.read_joint_positions(), "release")

        backend.step(config.settle_s)
        sponge_end = backend.get_body_pose("sponge")
        paper_end = backend.get_body_pose("paper")
        recorder.record(backend, backend.read_joint_positions(), "final")

    completion_time = backend.time_s - start_time
    success = _sponge_on_paper(sponge_end, paper_end)
    _write_outcome(
        out / PICK_AND_PLACE_OUTCOME,
        success=success,
        completion_time_s=completion_time,
        sponge_end=sponge_end,
        paper_end=paper_end,
        notes=notes,
    )
    return {
        "success": str(success).lower(),
        "completion_time_s": _format(completion_time),
    }


_SCENARIO_FUNCTIONS: Final[
    dict[
        str,
        Callable[
            [SimBackend, Path, SettleConfig, list[np.ndarray] | None], dict[str, str]
        ],
    ]
] = {
    "fk_sweep": _scenario_fk_sweep,
    "ik_reach": _scenario_ik_reach,
    "singularity": _scenario_singularity,
    "damped_ik": _scenario_damped_ik,
    "camera_to_base": _scenario_camera_to_base,
    "pick_and_place": _scenario_pick_and_place,
}


def run_scenario(
    name: str,
    out: Path,
    *,
    config: SettleConfig | None = None,
    width: int = 320,
    height: int = 240,
    frames: list[np.ndarray] | None = None,
) -> dict[str, str]:
    """Run one scenario headless and return a small summary mapping.

    ``name`` is one of :data:`SCENARIOS`. Raises ``ValueError`` for an unknown
    name. The CSV is written under ``out``; the directory is created if needed.

    When ``frames`` is a list, the ``top_cam`` image of every recorded frame is
    appended to it in CSV-row order, so the caller can write a video of the same
    run. Passing ``None`` (the default) skips rendering entirely.
    """
    if name not in _SCENARIO_FUNCTIONS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {', '.join(SCENARIOS)}")
    config = config or SettleConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with SimBackend(width=width, height=height) as backend:
        return _SCENARIO_FUNCTIONS[name](backend, out, config, frames)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, whose help text is the how-to."""
    parser = argparse.ArgumentParser(
        description=(
            "Run L03 goal-function demonstration scenarios in the MuJoCo "
            "sponge-to-paper scene and write per-frame CSV files."
        ),
        epilog=(
            "Each scenario writes <scenario>.csv under --out. Columns: frame, "
            "time_s, the six joints actually read back, our FK gripper pose, the "
            "MuJoCo gripperframe pose (metres plus qw,qx,qy,qz in (w,x,y,z) "
            "order, robot base frame), the Yoshikawa manipulability, and a "
            "free-text note. pick_and_place also writes "
            "pick_and_place_outcome.txt with success, completion_time_s, and the "
            "final sponge and paper positions. Rendering is off by default and "
            "the scenarios run headless."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
        help="which scenario to run (default: all)",
    )
    parser.add_argument(
        "--out",
        default="results/l03_sim",
        help="output directory for the CSV files (default: results/l03_sim)",
    )
    parser.add_argument(
        "--frame-dt",
        type=float,
        default=FRAME_DT,
        help=f"simulated seconds per control step (default: {FRAME_DT})",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=SETTLE_S,
        help=f"simulated seconds to settle before recording a frame (default: {SETTLE_S})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "short headless run: settle for 0.04 s per command so the script "
            "can be smoke-tested without rendering (overrides --settle-s)"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "also write an mp4 per scenario from that scenario's recorded top_cam "
            "frames, one per CSV row (needs OpenGL and imageio; off by default)"
        ),
    )
    parser.add_argument("--width", type=int, default=320, help="render width (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="render height (default: 240)")
    return parser


def _render_video(name: str, out: Path, frames: Sequence[np.ndarray]) -> None:
    """Write ``frames``, the recorded ``top_cam`` images of ``name``, to ``{name}.mp4``.

    One frame per logged CSV row, so the video shows that scenario's commanded
    trajectory. It is never called unless ``--render`` is given, and it raises
    ``ImportError`` when the video writer is unavailable instead of failing
    silently.
    """
    import imageio.v2 as imageio

    writer = imageio.get_writer(out / f"{name}.mp4", fps=VIDEO_FPS)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested scenarios and return an exit code."""
    arguments = _build_parser().parse_args(argv)
    out = Path(arguments.out)
    settle_s = 0.04 if arguments.smoke else arguments.settle_s
    config = SettleConfig(frame_dt=arguments.frame_dt, settle_s=settle_s)
    names = SCENARIOS if arguments.scenario == "all" else (arguments.scenario,)
    for name in names:
        start = time.time()
        frames: list[np.ndarray] | None = [] if arguments.render else None
        summary = run_scenario(
            name,
            out,
            config=config,
            width=arguments.width,
            height=arguments.height,
            frames=frames,
        )
        elapsed = time.time() - start
        print(f"[l03_demo] {name}: {elapsed:.1f}s {summary}")
        if frames is not None:
            _render_video(name, out, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
