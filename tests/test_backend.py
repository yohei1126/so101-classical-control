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
        frame = backend.get_camera_image("wrist_cam")

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.size > 0
    assert np.unique(frame).size > 1


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
