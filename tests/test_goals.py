"""Tests for the L01-L03 goal-level function layer.

Property IDs (P1-P14) refer to the L03 plan Page section
"L03 資料を踏まえた性質リスト（承認待ち案）"
(https://cursor.com/canvas/store-56725785-abdf-4ff8-831d-eb51fae6e592/ee0fe99b-a7a7-4f63-b17f-8059f8e116d1).
New numeric tolerances below are **provisional, pending Yohei approval**.
The MuJoCo comparison reuses the in-repo 1e-6 m / 1e-6 rad of
``tests/test_dh_fk.py`` and is not marked provisional.
"""

from __future__ import annotations

import inspect
import math

import mujoco
import numpy as np
import pytest

from so101_classical_control import goals
from so101_classical_control.calibration import (
    build_motion_pairs,
    camera_to_base,
    solve_ax_xb,
)
from so101_classical_control.goals import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    GraspPlan,
    UnreachableTarget,
    calibrate_wrist_camera_to_gripper,
    carry_to_paper,
    close_gripper_on_object,
    express_camera_point_in_base_frame,
    express_mujoco_camera_pose_in_opencv,
    keep_gripper_fixed_to_first_order,
    locate_gripper_from_joint_angles,
    locate_sponge_in_base_frame,
    locate_top_camera_in_base_frame,
    map_joint_velocity_to_gripper_twist,
    plan_grasp_pose,
    reach_target_pose,
    release_object,
    report_manipulability,
    tell_whether_planar_target_is_reachable,
)
from so101_classical_control.kinematics import (
    ARM_JOINT_NAMES,
    IK_TRANSLATION_TOL,
    adjoint,
    body_jacobian,
    fk_in_space,
    forward_kinematics,
    mujoco_camera_to_opencv,
    null_space_projector,
    pose,
    rx,
    ry,
    rz,
    solve_position_ik,
    space_jacobian,
    transform_point,
)
from so101_classical_control.kinematics.planar import planar_forward_kinematics, planar_jacobian
from so101_classical_control.sim.scene import scene_xml_path

# In-repo MuJoCo / DH comparison (tests/test_dh_fk.py). Not provisional.
TRANSLATION_TOL = 1e-6  # metres
ROTATION_TOL = 1e-6  # radians

# Provisional tolerances from the L03 plan Page, pending Yohei approval.
P1_FROBENIUS_TOL = 1e-10
P1_DET_TOL = 1e-10
P2_POSITION_TOL = 1e-12  # metres
P3_TRANSLATION_TOL = 1e-9  # metres
P3_ROTATION_TOL = 1e-9  # radians
P5_JACOBIAN_FROBENIUS_TOL = 1e-5
P5_FINITE_DIFF_EPS = 1e-6
P6_ADJOINT_FROBENIUS_TOL = 1e-9
P7_SINGULAR_W_TOL = 1e-12
P8_POSITION_TOL = 1e-12  # metres
P10_DISTINCT_ANGLE_TOL = 1e-9  # radians
P12_PROJECTOR_TOL = 1e-10

L1 = 1.0
L2 = 0.6  # L1 != L2 so the inner-radius case of P9 is nonempty
SEED = 20261120
ARM_LIMITS = (
    (-1.91986, 1.91986),
    (-1.7453293, 1.7453293),
    (-1.69, 1.69),
    (-1.658063, 1.658063),
    (-2.7438473, 2.8412063),
)


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    """Return the angle in radians of ``first.T @ second``.

    Uses the well-conditioned ``arctan2`` form from ``tests/test_handeye_axxb.py``;
    ``arccos((trace - 1) / 2)`` loses digits near zero and cannot resolve 1e-9.
    """
    relative = first.T @ second
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


def _assert_se3(transform: np.ndarray, label: str) -> None:
    """P1: rotation is near SO(3) and the last row is ``[0, 0, 0, 1]``."""
    assert transform.shape == (4, 4), label
    rotation = transform[:3, :3]
    residual = rotation.T @ rotation - np.eye(3)
    assert np.linalg.norm(residual, ord="fro") <= P1_FROBENIUS_TOL, label
    assert abs(float(np.linalg.det(rotation)) - 1.0) <= P1_DET_TOL, label
    np.testing.assert_allclose(
        transform[3, :], [0.0, 0.0, 0.0, 1.0], atol=P1_FROBENIUS_TOL, rtol=0.0
    )


def _sample_arm_angles(count: int) -> list[np.ndarray]:
    """Return ``count`` seeded configurations inside the XML joint ranges."""
    generator = np.random.default_rng(SEED)
    samples = [np.zeros(5), np.array([(low + high) / 2.0 for low, high in ARM_LIMITS])]
    for _ in range(count):
        samples.append(
            np.array([generator.uniform(low, high) for low, high in ARM_LIMITS])
        )
    return samples


# ---------------------------------------------------------------------------
# Docstring contract
# ---------------------------------------------------------------------------


def test_public_goal_functions_have_mr_lecture_means_docstrings() -> None:
    """Every public callable in goals has MR / Lecture / Means lines."""
    missing = []
    for name in goals.__all__:
        value = getattr(goals, name)
        if not callable(value) or isinstance(value, type):
            continue
        docstring = inspect.getdoc(value) or ""
        lines = [line.strip() for line in docstring.splitlines()]
        if not any(line.startswith("MR:") for line in lines):
            missing.append(f"{name}: MR:")
        if not any(line.startswith("Lecture:") for line in lines):
            missing.append(f"{name}: Lecture:")
        if not any(line.startswith("Means:") for line in lines):
            missing.append(f"{name}: Means:")
    assert missing == []


# ---------------------------------------------------------------------------
# P1, P2, P3, P4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _sample_arm_angles(5))
def test_p1_goal_fk_output_is_se3(q: np.ndarray) -> None:
    _assert_se3(locate_gripper_from_joint_angles(q), f"q={q}")


def test_p1_planar_fk_is_homogeneous() -> None:
    transform = planar_forward_kinematics([0.3, -0.4], L1, L2)
    assert transform.shape == (3, 3)
    rotation = transform[:2, :2]
    residual = rotation.T @ rotation - np.eye(2)
    assert np.linalg.norm(residual, ord="fro") <= P1_FROBENIUS_TOL
    assert abs(float(np.linalg.det(rotation)) - 1.0) <= P1_DET_TOL
    np.testing.assert_allclose(transform[2, :], [0.0, 0.0, 1.0], atol=P1_FROBENIUS_TOL)


def test_p2_planar_product_matches_trig_closed_form() -> None:
    for theta1, theta2 in ((0.0, 0.0), (0.4, -0.7), (math.pi / 2, math.pi / 3)):
        transform = planar_forward_kinematics([theta1, theta2], L1, L2)
        closed = np.array(
            [
                L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2),
                L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2),
            ]
        )
        np.testing.assert_allclose(transform[:2, 2], closed, atol=P2_POSITION_TOL, rtol=0.0)


@pytest.mark.parametrize("q", _sample_arm_angles(8))
def test_p3_dh_fk_matches_fk_in_space(q: np.ndarray) -> None:
    ours = locate_gripper_from_joint_angles(q)
    poe = fk_in_space(q)
    np.testing.assert_allclose(ours, forward_kinematics(q), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(ours[:3, 3], poe[:3, 3], atol=P3_TRANSLATION_TOL, rtol=0.0)
    assert _rotation_error(ours[:3, :3], poe[:3, :3]) <= P3_ROTATION_TOL


@pytest.mark.parametrize("case", ["zeros", "mid_range"])
def test_p4_goal_fk_matches_mujoco_gripperframe(case: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    q = np.zeros(len(ARM_JOINT_NAMES))
    if case == "mid_range":
        q = np.array([(low + high) / 2.0 for low, high in ARM_LIMITS])
    for name, value in zip(ARM_JOINT_NAMES, q, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)
    transform = locate_gripper_from_joint_angles(q)
    np.testing.assert_allclose(
        transform[:3, 3],
        data.site_xpos[site_id],
        atol=TRANSLATION_TOL,
        rtol=0.0,
    )
    rotation = data.site_xmat[site_id].reshape(3, 3)
    assert _rotation_error(transform[:3, :3], rotation) <= ROTATION_TOL


# ---------------------------------------------------------------------------
# P5, P6, P7
# ---------------------------------------------------------------------------


def test_p5_planar_jacobian_matches_finite_difference() -> None:
    angles = np.array([0.4, 0.7])
    analytical = planar_jacobian(angles, L1, L2)
    numeric = np.zeros((2, 2))
    base = planar_forward_kinematics(angles, L1, L2)[:2, 2]
    for index in range(2):
        stepped = angles.copy()
        stepped[index] += P5_FINITE_DIFF_EPS
        numeric[:, index] = (
            planar_forward_kinematics(stepped, L1, L2)[:2, 2] - base
        ) / P5_FINITE_DIFF_EPS
    assert np.linalg.norm(analytical - numeric, ord="fro") <= P5_JACOBIAN_FROBENIUS_TOL


def test_p5_so101_space_jacobian_matches_finite_difference() -> None:
    angles = np.array([0.2, -0.3, 0.5, 0.8, -0.4])
    analytical = space_jacobian(angles)
    # Finite-difference the end-effector position through the linear-velocity
    # part of the spatial Jacobian: v + omega x p.
    position = fk_in_space(angles)[:3, 3]
    predicted = analytical[3:, :] + np.cross(analytical[:3, :].T, position).T
    numeric = np.zeros((3, 5))
    for index in range(5):
        stepped = angles.copy()
        stepped[index] += P5_FINITE_DIFF_EPS
        numeric[:, index] = (fk_in_space(stepped)[:3, 3] - position) / P5_FINITE_DIFF_EPS
    assert np.linalg.norm(predicted - numeric, ord="fro") <= P5_JACOBIAN_FROBENIUS_TOL


def test_p6_space_and_body_jacobians_are_adjoint_related() -> None:
    angles = np.array([0.2, -0.3, 0.5, 0.8, -0.4])
    transform = fk_in_space(angles)
    space = space_jacobian(angles)
    body = body_jacobian(angles)
    converted = adjoint(transform) @ body
    assert np.linalg.norm(space - converted, ord="fro") <= P6_ADJOINT_FROBENIUS_TOL


def test_p7_planar_manipulability_vanishes_at_singularities() -> None:
    for theta2 in (0.0, math.pi):
        weight = report_manipulability([0.3, theta2], l1=L1, l2=L2)
        assert weight < P7_SINGULAR_W_TOL
    regular = report_manipulability([0.3, math.pi / 4], l1=L1, l2=L2)
    assert regular > 10.0 * P7_SINGULAR_W_TOL


# ---------------------------------------------------------------------------
# P8, P9, P10
# ---------------------------------------------------------------------------


def test_p8_p10_analytical_ik_round_trip_and_two_branches() -> None:
    point = np.array([0.8, 0.3])
    assert tell_whether_planar_target_is_reachable(point, L1, L2)
    both = reach_target_pose(point, l1=L1, l2=L2, elbow="both")
    assert isinstance(both, tuple) and len(both) == 2
    up, down = both
    assert abs(up[1] - down[1]) > P10_DISTINCT_ANGLE_TOL
    assert up[1] * down[1] < 0.0
    for angles in (up, down):
        tip = planar_forward_kinematics(angles, L1, L2)[:2, 2]
        np.testing.assert_allclose(tip, point, atol=P8_POSITION_TOL, rtol=0.0)


def test_p8_boundary_target_is_reachable() -> None:
    point = np.array([L1 + L2, 0.0])
    solution = reach_target_pose(point, l1=L1, l2=L2, elbow="up")
    assert not isinstance(solution, UnreachableTarget)
    tip = planar_forward_kinematics(solution, L1, L2)[:2, 2]
    np.testing.assert_allclose(tip, point, atol=P8_POSITION_TOL, rtol=0.0)


def test_p9_targets_outside_the_annulus_return_unreachable() -> None:
    far = reach_target_pose([L1 + L2 + 0.2, 0.0], l1=L1, l2=L2)
    inner = reach_target_pose([abs(L1 - L2) * 0.5, 0.0], l1=L1, l2=L2)
    assert isinstance(far, UnreachableTarget)
    assert isinstance(inner, UnreachableTarget)
    assert tell_whether_planar_target_is_reachable([L1 + L2 + 0.2, 0.0], L1, L2) is False
    assert tell_whether_planar_target_is_reachable([abs(L1 - L2) * 0.5, 0.0], L1, L2) is False


# ---------------------------------------------------------------------------
# P11, P12, P14
# ---------------------------------------------------------------------------


def test_p11_p14_position_ik_converges_without_increasing_error() -> None:
    seed = np.zeros(5)
    target = forward_kinematics([0.2, -0.3, 0.4, 0.8, 0.0])[:3, 3]
    start_error = float(np.linalg.norm(target - forward_kinematics(seed)[:3, 3]))
    result = solve_position_ik(target, seed)
    assert result.success
    assert result.position_error <= IK_TRANSLATION_TOL
    assert result.position_error <= start_error
    recovered = reach_target_pose(target, q0=seed)
    assert not isinstance(recovered, UnreachableTarget)
    final = locate_gripper_from_joint_angles(recovered)[:3, 3]
    assert float(np.linalg.norm(final - target)) <= IK_TRANSLATION_TOL


def test_p14_failed_ik_is_not_counted_as_success() -> None:
    far = np.array([10.0, 0.0, 0.0])
    result = reach_target_pose(far, q0=np.zeros(5))
    assert isinstance(result, UnreachableTarget)


def test_p12_null_space_projector_on_redundant_planar_3r() -> None:
    """A 2-task / 3-joint planar arm has a 1-D null space (not the 6x5 SO-101 J)."""
    t1, t2, t3 = 0.4, 0.7, -0.5
    l1 = l2 = l3 = 1.0
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    s123, c123 = math.sin(t1 + t2 + t3), math.cos(t1 + t2 + t3)
    jacobian = np.array(
        [
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
        ]
    )
    z = np.array([0.3, -0.4, 0.8])
    residual = jacobian @ (null_space_projector(jacobian, damping=0.0) @ z)
    scale = float(np.linalg.norm(z) + 1e-16)
    assert np.linalg.norm(residual) / scale < P12_PROJECTOR_TOL


# ---------------------------------------------------------------------------
# L02 wrappers and task-level composition
# ---------------------------------------------------------------------------


def test_l02_goal_wrappers_match_existing_means() -> None:
    transform = pose(rz(0.3), [0.1, -0.2, 0.4])
    point = np.array([0.05, 0.02, 0.01])
    np.testing.assert_allclose(
        express_camera_point_in_base_frame(transform, point),
        transform_point(transform, point),
    )
    base_marker = pose(np.eye(3), [0.2, 0.0, 0.1])
    marker_in_camera = pose(rz(0.1), [0.0, 0.05, 0.3])
    np.testing.assert_allclose(
        locate_top_camera_in_base_frame(base_marker, marker_in_camera),
        camera_to_base(base_marker, marker_in_camera),
    )
    np.testing.assert_allclose(
        express_mujoco_camera_pose_in_opencv(transform),
        mujoco_camera_to_opencv(transform),
    )


def test_calibrate_wrist_camera_to_gripper_matches_solver() -> None:
    x_gt = pose(rz(0.2), [0.01, 0.02, 0.03])
    # Varied rotation axes so the motion pairs are not collinear; four
    # configurations yield six pairs, enough for solve_ax_xb.
    hand = [
        pose(np.eye(3), [0.0, 0.0, 0.0]),
        pose(rz(0.6), [0.1, 0.0, 0.0]),
        pose(rx(0.5), [0.0, 0.1, 0.0]),
        pose(ry(0.4), [0.0, 0.0, 0.1]),
    ]
    camera = [item @ x_gt for item in hand]
    recovered = calibrate_wrist_camera_to_gripper(hand, camera)
    a_list, b_list = build_motion_pairs(hand, camera)
    expected = solve_ax_xb(a_list, b_list)
    np.testing.assert_allclose(recovered, expected)


def test_locate_sponge_composes_camera_to_base_mapping() -> None:
    camera = pose(rz(0.2), [0.15, -0.35, 0.6])
    point = np.array([0.02, 0.03, 0.4])
    np.testing.assert_allclose(
        locate_sponge_in_base_frame(point, T_base_cam=camera),
        express_camera_point_in_base_frame(camera, point),
    )


def test_plan_grasp_pose_is_top_down() -> None:
    sponge = pose(np.eye(3), [0.30, 0.08, 0.015])
    plan = plan_grasp_pose(sponge)
    assert isinstance(plan, GraspPlan)
    finger_world = plan.pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(finger_world, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(plan.approach_direction, [0.0, 0.0, 1.0])


def test_gripper_commands_are_inside_the_documented_range() -> None:
    low, high = -0.174533, 1.7453292
    assert low <= close_gripper_on_object() <= high
    assert low <= release_object() <= high
    assert close_gripper_on_object() == GRIPPER_CLOSED_RAD
    assert release_object() == GRIPPER_OPEN_RAD
    assert close_gripper_on_object() < release_object()


def test_carry_to_paper_returns_typed_result() -> None:
    paper = pose(np.eye(3), [0.30, -0.08, 0.001])
    result = carry_to_paper(paper, q0=np.zeros(5), clearance=0.0)
    assert isinstance(result, UnreachableTarget | np.ndarray)


def test_invalid_goal_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        locate_gripper_from_joint_angles([0.0, 0.0])
    with pytest.raises(ValueError):
        reach_target_pose([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        tell_whether_planar_target_is_reachable([0.0, 0.0], 0.0, 1.0)
    with pytest.raises(ValueError):
        map_joint_velocity_to_gripper_twist(np.zeros(5), [1.0])
    with pytest.raises(ValueError):
        keep_gripper_fixed_to_first_order(np.zeros(5), np.ones(3))
    with pytest.raises(ValueError):
        locate_sponge_in_base_frame([0.0, 0.0, 0.0])
