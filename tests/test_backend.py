"""Tests for the shared robot backend interface and its sim/real implementations."""

import mujoco
import numpy as np
import pytest

from so101_classical_control.backend import JOINT_NAMES, RobotBackend
from so101_classical_control.real import RealBackend
from so101_classical_control.sim import SimBackend
from so101_classical_control.sim.scene import scene_xml_path

# Position actuators in the MJCF use high gains (kp ~ 998).  After one second
# of simulation every joint tracks its target to well under 1e-3 rad, so 0.01 rad
# (~0.57 deg) leaves a realistic margin for small gravity/contact effects while
# still failing if commands are not applied.
SETTLE_TOLERANCE_RAD = 0.01

REACHABLE_TARGETS = {
    "shoulder_pan": 0.30,
    "shoulder_lift": 0.20,
    "elbow_flex": -0.30,
    "wrist_flex": 0.15,
    "wrist_roll": 0.40,
    "gripper": 0.60,
}


def _mjcf_joint_ranges() -> dict[str, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    ranges: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1
        low, high = model.jnt_range[joint_id]
        ranges[name] = (float(low), float(high))
    return ranges


def _mjcf_ctrl_ranges() -> dict[str, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    ranges: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert actuator_id != -1
        low, high = model.actuator_ctrlrange[actuator_id]
        ranges[name] = (float(low), float(high))
    return ranges


def test_sim_backend_reads_all_joints_within_mjcf_ranges() -> None:
    ranges = _mjcf_joint_ranges()

    with SimBackend() as backend:
        assert isinstance(backend, RobotBackend)
        positions = backend.read_joint_positions()

        assert set(positions) == set(JOINT_NAMES)
        for name, value in positions.items():
            assert np.isfinite(value), name
            low, high = ranges[name]
            assert low <= value <= high, (name, value, low, high)


def test_sim_backend_command_settles_and_reads_back() -> None:
    with SimBackend() as backend:
        backend.command_joint_targets(REACHABLE_TARGETS)
        backend.step(1.0)
        positions = backend.read_joint_positions()

        assert backend.time_s > 0.0
        for name, target in REACHABLE_TARGETS.items():
            assert positions[name] == pytest.approx(target, abs=SETTLE_TOLERANCE_RAD), name


def test_sim_backend_reads_wrist_camera_headless() -> None:
    with SimBackend(width=320, height=240) as backend:
        # Rendering before the first step exercises mj_forward at t = 0.
        frame = backend.get_camera_image("wrist_cam")
        backend.step(0.1)
        post_step_frame = backend.get_camera_image("wrist_cam")

    for image in (frame, post_step_frame):
        assert isinstance(image, np.ndarray)
        assert image.shape == (240, 320, 3)
        assert image.dtype == np.uint8
        assert image.size > 0
        assert np.unique(image).size > 1


def test_sim_backend_rejects_unknown_joint() -> None:
    with SimBackend() as backend:
        with pytest.raises(KeyError, match="unknown joint"):
            backend.command_joint_targets({"does_not_exist": 0.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sim_backend_rejects_non_finite_targets(value: float) -> None:
    with SimBackend() as backend:
        with pytest.raises(ValueError, match="finite"):
            backend.command_joint_targets({"shoulder_pan": value})


def test_sim_backend_rejects_target_outside_ctrlrange() -> None:
    high = _mjcf_ctrl_ranges()["elbow_flex"][1]

    with SimBackend() as backend:
        with pytest.raises(ValueError, match="ctrlrange"):
            backend.command_joint_targets({"elbow_flex": high + 0.5})


def test_sim_backend_partial_command_preserves_omitted_targets() -> None:
    with SimBackend() as backend:
        backend.command_joint_targets(REACHABLE_TARGETS)
        backend.command_joint_targets({"gripper": 0.90})
        backend.step(1.0)
        positions = backend.read_joint_positions()

    for name, target in REACHABLE_TARGETS.items():
        expected = 0.90 if name == "gripper" else target
        assert positions[name] == pytest.approx(expected, abs=SETTLE_TOLERANCE_RAD), name


@pytest.mark.parametrize(
    "invalid_targets",
    [
        {"shoulder_pan": 0.1, "does_not_exist": 0.1},
        {"shoulder_pan": 0.1, "elbow_flex": float("nan")},
        {"shoulder_pan": 0.1, "elbow_flex": 99.0},
    ],
)
def test_sim_backend_validates_all_targets_before_mutating(
    invalid_targets: dict[str, float],
) -> None:
    with SimBackend() as backend:
        backend.command_joint_targets(REACHABLE_TARGETS)
        with pytest.raises((KeyError, ValueError)):
            backend.command_joint_targets(invalid_targets)
        # The failed mapping must not have applied any of its entries; the
        # previously accepted targets are still what the arm settles to.
        backend.step(1.0)
        positions = backend.read_joint_positions()

    for name, target in REACHABLE_TARGETS.items():
        assert positions[name] == pytest.approx(target, abs=SETTLE_TOLERANCE_RAD), name


@pytest.mark.parametrize(
    "duration_s",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_sim_backend_step_rejects_invalid_durations(duration_s: float) -> None:
    with SimBackend() as backend:
        with pytest.raises(ValueError):
            backend.step(duration_s)
        assert backend.time_s == 0.0


def test_sim_backend_step_advances_time() -> None:
    with SimBackend() as backend:
        backend.step(1.0)
        assert backend.time_s == pytest.approx(1.0)


def test_sim_backend_step_zero_is_noop() -> None:
    with SimBackend() as backend:
        backend.step(1.0)
        before = backend.time_s
        backend.step(0)
        assert backend.time_s == before


def test_sim_backend_methods_fail_after_close() -> None:
    backend = SimBackend()
    backend.close()
    backend.close()  # idempotent, must not raise

    with pytest.raises(RuntimeError, match="closed"):
        backend.read_joint_positions()
    with pytest.raises(RuntimeError, match="closed"):
        backend.command_joint_targets({"gripper": 0.5})
    with pytest.raises(RuntimeError, match="closed"):
        backend.step(0.1)
    with pytest.raises(RuntimeError, match="closed"):
        _ = backend.time_s
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_camera_image("wrist_cam")


def test_sim_backend_rejects_unknown_camera() -> None:
    with SimBackend() as backend:
        with pytest.raises(KeyError, match="unknown camera"):
            backend.get_camera_image("does_not_exist")


def test_sim_backend_camera_fails_after_close() -> None:
    backend = SimBackend()
    backend.close()

    with pytest.raises(RuntimeError, match="closed"):
        backend.get_camera_image("wrist_cam")


def test_real_backend_interface_methods_are_unimplemented() -> None:
    backend = RealBackend()

    assert isinstance(backend, RobotBackend)
    with pytest.raises(NotImplementedError):
        backend.read_joint_positions()
    with pytest.raises(NotImplementedError):
        backend.command_joint_targets(REACHABLE_TARGETS)
    with pytest.raises(NotImplementedError):
        backend.get_camera_image("wrist_cam")
    with pytest.raises(NotImplementedError):
        backend.step(0.1)
    with pytest.raises(NotImplementedError):
        _ = backend.time_s


def test_real_backend_close_and_context_manager_succeed() -> None:
    with RealBackend() as backend:
        assert isinstance(backend, RobotBackend)

    RealBackend().close()
