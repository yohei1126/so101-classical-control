"""Tests for the SO-101 standard-DH forward kinematics.

The DH model is compared against the MuJoCo ``gripperframe`` site pose, read
directly from ``data.site_xpos`` and ``data.site_xmat`` after setting the five
arm joint positions and calling ``mj_forward``. It never goes through
``SimBackend`` and no site accessor is added to the backend.

Tolerances: 1e-6 m for translation and 1e-6 rad for rotation, where the
rotation error is the angle of ``R_dh.T @ R_mujoco``, computed as
``arccos(clip((trace(R_err) - 1) / 2, -1, 1))``.

Random configurations come from one seeded generator (``SEED``) and are
sampled uniformly inside each arm joint's ``model.jnt_range``. The gripper DoF
stays at zero for the random set.

The gripper DoF is excluded from arm FK: ``gripperframe`` is a child of the
``gripper`` body and a sibling of ``moving_jaw_so101_v1``, so moving the
``gripper`` joint must not move the site.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mujoco
import numpy as np
import pytest

from so101_classical_control.backend import JOINT_NAMES
from so101_classical_control.kinematics import (
    ARM_JOINT_NAMES,
    SO101_DH_PARAMS,
    SO101_T_BASE_TO_0,
    SO101_T_N_TO_EE,
    dh_transform,
    forward_kinematics,
)
from so101_classical_control.sim.scene import scene_xml_path

SEED = 20261114
RANDOM_SAMPLES = 200
TRANSLATION_TOL = 1e-6  # metres
ROTATION_TOL = 1e-6  # radians
NEAR_LIMIT_MARGIN = 1e-3  # radians inside a joint stop


class Scene(NamedTuple):
    """A loaded scene plus the indices needed to pose the arm."""

    model: mujoco.MjModel
    data: mujoco.MjData
    site_id: int
    arm_qpos: tuple[int, ...]
    gripper_qpos: int
    arm_ranges: tuple[tuple[float, float], ...]
    gripper_range: tuple[float, float]


@pytest.fixture(scope="module")
def scene() -> Scene:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    arm_qpos = []
    arm_ranges = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        arm_qpos.append(int(model.jnt_qposadr[joint_id]))
        low, high = model.jnt_range[joint_id]
        arm_ranges.append((float(low), float(high)))
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_low, gripper_high = model.jnt_range[gripper_id]
    gripper_range = (float(gripper_low), float(gripper_high))

    return Scene(
        model=model,
        data=data,
        site_id=site_id,
        arm_qpos=tuple(arm_qpos),
        gripper_qpos=int(model.jnt_qposadr[gripper_id]),
        arm_ranges=tuple(arm_ranges),
        gripper_range=gripper_range,
    )


def _rotation_error(r_dh: np.ndarray, r_mujoco: np.ndarray) -> float:
    """Return the angle in radians of ``R_dh.T @ R_mujoco``."""
    relative = r_dh.T @ r_mujoco
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _site_pose(
    scene: Scene, q: np.ndarray, gripper: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pose ``(R, p)`` of ``gripperframe`` after setting ``q`` and ``gripper``."""
    data = scene.data
    data.qpos[:] = scene.model.qpos0
    for address, value in zip(scene.arm_qpos, q, strict=True):
        data.qpos[address] = value
    data.qpos[scene.gripper_qpos] = gripper
    mujoco.mj_forward(scene.model, data)
    rotation = data.site_xmat[scene.site_id].reshape(3, 3).copy()
    translation = data.site_xpos[scene.site_id].copy()
    return rotation, translation


def _assert_matches(
    transform: np.ndarray, r_mujoco: np.ndarray, p_mujoco: np.ndarray, label: str
) -> None:
    """Assert the DH pose matches MuJoCo at 1e-6 m / 1e-6 rad."""
    np.testing.assert_allclose(
        transform[:3, 3],
        p_mujoco,
        atol=TRANSLATION_TOL,
        rtol=0.0,
        err_msg=f"translation mismatch for {label}",
    )
    error = _rotation_error(transform[:3, :3], r_mujoco)
    assert error <= ROTATION_TOL, f"rotation error {error:.3e} rad for {label}"


def _compare(scene: Scene, q: np.ndarray, label: str, gripper: float = 0.0) -> None:
    r_mujoco, p_mujoco = _site_pose(scene, q, gripper)
    _assert_matches(forward_kinematics(q), r_mujoco, p_mujoco, label)


def _mid_range(scene: Scene) -> np.ndarray:
    return np.array([(low + high) / 2.0 for low, high in scene.arm_ranges])


def _near_limits(scene: Scene, end: str) -> np.ndarray:
    if end == "lower":
        return np.array([low + NEAR_LIMIT_MARGIN for low, _ in scene.arm_ranges])
    return np.array([high - NEAR_LIMIT_MARGIN for _, high in scene.arm_ranges])


# ---------------------------------------------------------------------------
# FK against MuJoCo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["zeros", "mid_range", "near_lower", "near_upper"])
def test_fk_matches_mujoco_named_configurations(scene: Scene, case: str) -> None:
    if case == "zeros":
        q = np.zeros(len(ARM_JOINT_NAMES))
    elif case == "mid_range":
        q = _mid_range(scene)
    elif case == "near_lower":
        q = _near_limits(scene, "lower")
    else:
        q = _near_limits(scene, "upper")
    _compare(scene, q, case)


def test_fk_matches_mujoco_for_random_configurations(scene: Scene) -> None:
    rng = np.random.default_rng(SEED)
    for index in range(RANDOM_SAMPLES):
        q = np.array([rng.uniform(low, high) for low, high in scene.arm_ranges])
        _compare(scene, q, f"random sample {index}")


@pytest.mark.parametrize("gripper", [-0.174533, 0.0, 0.5, 1.7453292])
def test_gripper_motion_does_not_change_tool_pose(scene: Scene, gripper: float) -> None:
    q = _mid_range(scene)
    reference = forward_kinematics(q)

    # FK ignores the gripper value, explicitly or through a six-vector.
    np.testing.assert_array_equal(forward_kinematics([*q, gripper]), reference)
    mapping = dict(zip(ARM_JOINT_NAMES, q, strict=True))
    np.testing.assert_array_equal(forward_kinematics({**mapping, "gripper": gripper}), reference)

    # The MuJoCo site pose is unchanged too.
    r_reference, p_reference = _site_pose(scene, q, 0.0)
    r_moved, p_moved = _site_pose(scene, q, gripper)
    np.testing.assert_allclose(p_moved, p_reference, atol=TRANSLATION_TOL, rtol=0.0)
    assert _rotation_error(r_reference, r_moved) <= ROTATION_TOL


# ---------------------------------------------------------------------------
# Input forms and output contract
# ---------------------------------------------------------------------------


def test_six_vector_and_mapping_forms_match_five_vector(scene: Scene) -> None:
    q = _mid_range(scene)
    expected = forward_kinematics(q)
    mapping = dict(zip(ARM_JOINT_NAMES, q, strict=True))

    np.testing.assert_array_equal(forward_kinematics([*q, 0.7]), expected)
    np.testing.assert_array_equal(forward_kinematics(mapping), expected)
    np.testing.assert_array_equal(forward_kinematics({**mapping, "gripper": 0.7}), expected)


def test_output_contract() -> None:
    transform = forward_kinematics(np.zeros(len(ARM_JOINT_NAMES)))
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float64
    np.testing.assert_array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_arm_joint_names_are_prefix_of_joint_names() -> None:
    assert ARM_JOINT_NAMES == JOINT_NAMES[:5]
    assert "gripper" not in ARM_JOINT_NAMES


def test_frozen_constants_have_the_expected_shapes() -> None:
    assert len(SO101_DH_PARAMS) == len(ARM_JOINT_NAMES)
    assert all(len(row) == 4 for row in SO101_DH_PARAMS)
    np.testing.assert_array_equal(SO101_T_BASE_TO_0.shape, (4, 4))
    np.testing.assert_array_equal(SO101_T_N_TO_EE.shape, (4, 4))
    # The pan axis turns about world -Z at q = 0, so the base rotation is Rx(pi).
    np.testing.assert_allclose(SO101_T_BASE_TO_0[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=0.0)


# ---------------------------------------------------------------------------
# dh_transform
# ---------------------------------------------------------------------------


def test_dh_transform_matches_l02_matrix() -> None:
    a, alpha, d, theta = 0.3, 0.4, 0.2, 0.5
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    expected = np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(dh_transform(a, alpha, d, theta), expected, atol=1e-15, rtol=0.0)


def test_dh_transform_reduces_to_elementary_terms() -> None:
    # a = d = alpha = 0: pure rotation about z.
    np.testing.assert_allclose(
        dh_transform(0.0, 0.0, 0.0, 0.7)[:3, :3],
        [
            [math.cos(0.7), -math.sin(0.7), 0.0],
            [math.sin(0.7), math.cos(0.7), 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-15,
    )
    # theta = 0 and a = d = 0: pure rotation about x.
    np.testing.assert_allclose(
        dh_transform(0.0, -math.pi / 2.0, 0.0, 0.0)[:3, :3],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (np.nan, 0.0, 0.0, 0.0),
        (0.0, np.inf, 0.0, 0.0),
        (0.0, 0.0, -np.inf, 0.0),
        (0.0, 0.0, 0.0, np.nan),
        ([0.0, 1.0], 0.0, 0.0, 0.0),
        (0.0, [0.0], 0.0, 0.0),
        (None, 0.0, 0.0, 0.0),
        (0.0, 0.0, "not a number", 0.0),
    ],
)
def test_dh_transform_rejects_invalid_inputs(arguments) -> None:
    with pytest.raises(ValueError):
        dh_transform(*arguments)


# ---------------------------------------------------------------------------
# Invalid configurations
# ---------------------------------------------------------------------------


def _full_mapping() -> dict[str, float]:
    return {name: 0.0 for name in ARM_JOINT_NAMES}


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(4),
        np.zeros(7),
        np.zeros((5, 1)),
        [0.0, 0.0, 0.0, 0.0, np.nan],
        ["0.0", "0.0", "0.0", "0.0", "not a number"],
        {"shoulder_pan": 0.0},
        {**_full_mapping(), "elbow": 0.0},
        {**_full_mapping(), "base": 0.0},
    ],
)
def test_forward_kinematics_rejects_invalid_configurations(q) -> None:
    with pytest.raises(ValueError):
        forward_kinematics(q)


# ---------------------------------------------------------------------------
# The frozen geometry is not an idealised textbook arm
# ---------------------------------------------------------------------------


def test_q0_keeps_the_y_offset_and_the_tool_tilt(scene: Scene) -> None:
    q = np.zeros(len(ARM_JOINT_NAMES))
    rotation, translation = _site_pose(scene, q)
    transform = forward_kinematics(q)
    _assert_matches(transform, rotation, translation, "q = 0")

    # The vendored model keeps the 18.3 mm Y offset of upper_arm, lower_arm and
    # wrist instead of collapsing them onto y = 0.
    for name in ("upper_arm", "lower_arm", "wrist"):
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.isclose(scene.data.xpos[body_id][1], -0.0182778, atol=1e-6)

    # Consequently the q = 0 tool point is off the XZ plane of a planar arm.
    assert abs(transform[1, 3]) > TRANSLATION_TOL
    assert np.isclose(transform[1, 3], translation[1], atol=TRANSLATION_TOL)

    # The gripper mount leaves a ~2.8 degree tilt about X, not an identity tool.
    tilt = _rotation_error(np.eye(3), transform[:3, :3])
    assert np.isclose(tilt, math.radians(2.8), atol=math.radians(0.05))
    assert not np.allclose(transform[:3, :3], np.eye(3), atol=1e-3)
