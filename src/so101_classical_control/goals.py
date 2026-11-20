"""Goal-level kinematics, calibration, and pick-and-place helpers.

The public names here say what a call achieves on the robot ("locate the
gripper", "reach a pose", "carry to the paper") rather than which algorithm
runs. Each function is thin: it validates its inputs and calls an existing mean
in :mod:`so101_classical_control.kinematics`,
:mod:`so101_classical_control.calibration`, or ``modern_robotics``.

Every public function has a docstring of this form::

    \"\"\"One-sentence English goal description.

    MR: §x.y, Eq. (n)
    Lecture: L0n (topic)
    Means: module.function
    \"\"\"

The ``MR`` and ``Lecture`` lines cite the verified sources for that goal; where
no numbered equation exists, the line says so instead of inventing one.

Unreachable targets are values, not exceptions: :func:`reach_target_pose`
returns :class:`UnreachableTarget` rather than raising, so a caller can record
"no solution" and leave the robot where it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .backend import JOINT_NAMES
from .calibration import build_motion_pairs, camera_to_base, solve_ax_xb
from .kinematics import (
    ARM_JOINT_NAMES,
    SO101_DH_PARAMS,
    SO101_T_BASE_TO_0,
    SO101_T_N_TO_EE,
    forward_kinematics,
    mujoco_camera_to_opencv,
    transform_point,
)
from .kinematics.dh import arm_joint_angles
from .kinematics.inverse import null_space_projector, solve_pose_ik, solve_position_ik
from .kinematics.planar import (
    planar_ik_solutions,
    planar_inverse_kinematics,
    planar_manipulability,
    planar_reachable,
)
from .kinematics.poe import space_jacobian

__all__ = [
    "GRASP_CLEARANCE",
    "GRASP_FINGER_AXIS",
    "GRASP_JAW_OPENING_AXIS",
    "GRASP_PINCH_OFFSET",
    "GRIPPER_CLOSED_RAD",
    "GRIPPER_OPEN_RAD",
    "SO101_WORKSPACE_RADIUS_BOUND",
    "GraspPlan",
    "UnreachableTarget",
    "calibrate_wrist_camera_to_gripper",
    "carry_to_paper",
    "close_gripper_on_object",
    "express_camera_point_in_base_frame",
    "express_mujoco_camera_pose_in_opencv",
    "keep_gripper_fixed_to_first_order",
    "locate_gripper_from_joint_angles",
    "locate_sponge_in_base_frame",
    "locate_top_camera_in_base_frame",
    "map_joint_velocity_to_gripper_twist",
    "plan_grasp_pose",
    "reach_target_pose",
    "release_object",
    "report_manipulability",
    "tell_whether_planar_target_is_reachable",
]

_DTYPE = np.float64
#: Number of joints in the arm chain.
_ARM_DOF = len(ARM_JOINT_NAMES)


def _workspace_radius_bound() -> float:
    """Return an upper bound on the distance from the base to ``gripperframe``.

    Each link translation of the DH chain contributes at most its own norm,
    because a rotation preserves length; summing them (plus the fixed base and
    tool offsets) bounds the reachable radius from above. It is a workspace fact
    of the model, not a course threshold.
    """
    terms = [float(np.linalg.norm(SO101_T_BASE_TO_0[:3, 3]))]
    for a, _alpha, d, _theta in SO101_DH_PARAMS:
        terms.append(float(np.linalg.norm(np.array([a, 0.0, d]))))
    terms.append(float(np.linalg.norm(SO101_T_N_TO_EE[:3, 3])))
    return float(sum(terms))


#: Upper bound on the reachable radius, in metres (see :func:`_workspace_radius_bound`).
SO101_WORKSPACE_RADIUS_BOUND: Final[float] = _workspace_radius_bound()

#: Gripper command, radians, that closes the jaws on an object. The hinge range
#: is ``[-0.174533, 1.7453292]`` and about ``0 rad`` is closed; this is a
#: conservative in-range value, not a verified jaw preload.
GRIPPER_CLOSED_RAD: Final[float] = 0.0
#: Gripper command, radians, that opens the jaws. Positive angles open the moving
#: jaw; this is a conservative in-range value, not a verified grasp width.
GRIPPER_OPEN_RAD: Final[float] = 0.9

#: Tool axis the fingers extend along, measured on the vendored gripper geometry
#: in ``gripperframe``. A top-down grasp points this axis along world ``-Z``.
GRASP_FINGER_AXIS: Final[np.ndarray] = np.array([1.0, 0.0, 0.0], dtype=_DTYPE)
#: Tool axis the jaws open along, measured on the vendored gripper geometry.
GRASP_JAW_OPENING_AXIS: Final[np.ndarray] = np.array([0.0, 0.0, 1.0], dtype=_DTYPE)
#: Midpoint between the jaw tips in ``gripperframe``, measured on the vendored
#: gripper geometry at ``q = 0``.
GRASP_PINCH_OFFSET: Final[np.ndarray] = np.array([0.0007, 0.0001, -0.0119], dtype=_DTYPE)
#: Pre-grasp clearance, metres, straight up from the grasp pose.
GRASP_CLEARANCE: Final[float] = 0.05


@dataclass(frozen=True)
class UnreachableTarget:
    """Typed "no solution" result returned instead of raising.

    Attributes
    ----------
    reason
        Short English explanation, e.g. ``"position outside the workspace bound"``.
    """

    reason: str


@dataclass(frozen=True)
class GraspPlan:
    """A geometric top-down grasp plan, in the robot base frame.

    Attributes
    ----------
    pose
        ``(4, 4)`` ``gripperframe`` target that puts the jaw midpoint on the
        grasp point with the fingers descending along world ``-Z``.
    pregrasp_pose
        ``(4, 4)`` of the same orientation lifted :data:`GRASP_CLEARANCE`
        straight up.
    approach_direction
        ``(3,)`` unit vector from the object toward the gripper, world ``+Z``.
    """

    pose: np.ndarray
    pregrasp_pose: np.ndarray
    approach_direction: np.ndarray


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_seed(value: object) -> np.ndarray:
    """Return ``value`` as the five arm angles accepted by the forward kinematics."""
    if value is None:
        return np.zeros(_ARM_DOF, dtype=_DTYPE)
    return np.asarray(arm_joint_angles(value), dtype=_DTYPE)


def _as_pose(value: object, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(4, 4)`` float64 homogeneous transform."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float64 array") from error
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_vector(value: object, size: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(size,)`` float64 array."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float64 array") from error
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_joint_velocity(value: object) -> np.ndarray:
    """Return a joint-velocity vector of length five, ignoring a trailing gripper."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError("joint_velocity must be convertible to a float64 array") from error
    if array.shape == (len(JOINT_NAMES),):
        array = array[:_ARM_DOF]
    elif array.shape != (_ARM_DOF,):
        raise ValueError(
            f"joint_velocity must have {_ARM_DOF} entries "
            f"({len(JOINT_NAMES)} with a trailing gripper entry), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("joint_velocity must contain only finite values")
    return array


def _top_down_rotation(yaw: float) -> np.ndarray:
    """Return the tool rotation with fingers down and jaws opening at ``yaw``.

    The finger axis (:data:`GRASP_FINGER_AXIS`, tool ``+X``) maps to world
    ``-Z``, so the fingers descend onto the object from above. The jaw opening
    axis (:data:`GRASP_JAW_OPENING_AXIS`, tool ``+Z``) is horizontal and turns
    with ``yaw`` about world ``+Z``. The remaining column is ``z x x`` so the
    result is a proper rotation.
    """
    finger = np.array([0.0, 0.0, -1.0], dtype=_DTYPE)
    opening = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=_DTYPE)
    side = np.cross(opening, finger)
    side = side / np.linalg.norm(side)
    return np.column_stack([finger, side, opening])


def _grasp_pose(position: np.ndarray, yaw: float) -> np.ndarray:
    """Return the ``gripperframe`` pose whose jaw midpoint sits at ``position``."""
    rotation = _top_down_rotation(yaw)
    transform = np.eye(4, dtype=_DTYPE)
    transform[:3, :3] = rotation
    transform[:3, 3] = position - rotation @ GRASP_PINCH_OFFSET
    return transform


def _lifted(transform: np.ndarray, height: float) -> np.ndarray:
    """Return ``transform`` translated ``height`` metres along world ``+Z``."""
    lifted = transform.copy()
    lifted[2, 3] += height
    return lifted


# ---------------------------------------------------------------------------
# L03 kinematics goals
# ---------------------------------------------------------------------------


def locate_gripper_from_joint_angles(joint_angles) -> np.ndarray:
    """Return the ``gripperframe`` pose in the robot base frame from joint angles.

    ``joint_angles`` accepts the forms of ``forward_kinematics``: five arm
    angles, six with a trailing gripper value that is ignored, or a mapping keyed
    by joint name.

    MR: §4.1, Eq. (4.14). Chapter 4 reaches the same goal (FK) via PoE; this
    wrapper uses the in-repo standard-DH ``forward_kinematics`` (L02 exercise
    sections 8-9), not ``FKinSpace``.
    Lecture: L03 (slide 3.3, exercise 1.2); L02 exercise sections 8-9
    Means: so101_classical_control.kinematics.forward_kinematics
    """
    return np.asarray(forward_kinematics(joint_angles), dtype=_DTYPE)


def reach_target_pose(
    target,
    *,
    q0=None,
    l1: float | None = None,
    l2: float | None = None,
    planar: bool | None = None,
    elbow: str = "up",
    joint_limits: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | UnreachableTarget:
    """Return joint angles that reach ``target``, or an ``UnreachableTarget``.

    Two dispatches are supported:

    - Planar 2R: pass ``planar=True`` (or a two-vector target together with
      ``l1`` and ``l2``). ``target`` is ``(x, y)`` metres; ``elbow`` is
      ``"up"``, ``"down"``, or ``"both"``. ``"both"`` returns ``(up, down)`` as a
      tuple. The solve is the closed-form 2R law, and unreachable targets return
      ``UnreachableTarget`` rather than raising.
    - SO-101: ``target`` is a ``(4, 4)`` pose or a ``(3,)`` position for
      ``gripperframe``; ``q0`` seeds the numerical IK. A position farther than
      :data:`SO101_WORKSPACE_RADIUS_BOUND`, or a solve whose forward kinematics
      misses the stop condition, returns ``UnreachableTarget``.

    MR: §6.1, Eq. (6.1); §6.2, Eq. (6.6). The damped term is not in MR chapter 6
    (no numbered MR equation for damping).
    Lecture: L03 (slide 3.5, exercises 3.1-3.3)
    Means: so101_classical_control.kinematics.planar.planar_inverse_kinematics or
    so101_classical_control.kinematics.inverse.solve_position_ik / solve_pose_ik
    """
    use_planar = planar if planar is not None else (l1 is not None and l2 is not None)
    if use_planar:
        if l1 is None or l2 is None:
            raise ValueError("planar IK needs both l1 and l2")
        point = _as_vector(target, 2, "target")
        if elbow == "both":
            solutions = planar_ik_solutions(point, l1, l2)
            if solutions is None:
                return UnreachableTarget(reason="planar target outside the annular workspace")
            return solutions
        solution = planar_inverse_kinematics(point, l1, l2, elbow=elbow)
        if solution is None:
            return UnreachableTarget(reason="planar target outside the annular workspace")
        return solution

    target_array = np.asarray(target, dtype=_DTYPE)
    if target_array.shape == (4, 4):
        if not np.all(np.isfinite(target_array)):
            raise ValueError("target must contain only finite values")
        distance = float(np.linalg.norm(target_array[:3, 3]))
    elif target_array.shape == (3,):
        if not np.all(np.isfinite(target_array)):
            raise ValueError("target must contain only finite values")
        distance = float(np.linalg.norm(target_array))
    else:
        raise ValueError(
            "target must be a planar (2,), cartesian (3,), or pose (4, 4) array, "
            f"got {target_array.shape}"
        )

    if distance > SO101_WORKSPACE_RADIUS_BOUND:
        return UnreachableTarget(reason="position outside the workspace radius bound")

    seed = _as_seed(q0)
    if target_array.shape == (3,):
        result = solve_position_ik(target_array, seed, joint_limits=joint_limits)
    else:
        result = solve_pose_ik(target_array, seed, joint_limits=joint_limits)
    if not result.success:
        return UnreachableTarget(reason=result.reason)
    return result.joint_angles


def map_joint_velocity_to_gripper_twist(joint_angles, joint_velocity) -> np.ndarray:
    """Return the ``(6,)`` spatial twist ``J_s qdot`` of the gripper.

    The first three components are angular velocity and the last three the
    velocity of the ``gripperframe`` origin, both in the space (base) frame. The
    spatial Jacobian is ``6 x 5``; ``joint_velocity`` has length five, or six
    with a trailing gripper entry that is ignored.

    MR: §5.1, Eq. (5.10), (5.17)
    Lecture: L03 (slide 3.4, exercise 2.1)
    Means: modern_robotics.JacobianSpace via
    so101_classical_control.kinematics.poe.space_jacobian
    """
    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    velocity = _as_joint_velocity(joint_velocity)
    return np.asarray(space_jacobian(angles) @ velocity, dtype=_DTYPE)


def keep_gripper_fixed_to_first_order(
    joint_angles, secondary_joint_velocity, *, damping: float = 0.0
) -> np.ndarray:
    """Return the ``(5,)`` null-space part ``(I - J^+ J) z`` of a joint velocity.

    ``J`` is the ``6 x 5`` spatial Jacobian and ``J^+`` its (damped, when
    ``damping > 0``) pseudo-inverse. With ``damping == 0`` at full rank the
    projector is exact, so the result produces no first-order gripper twist.
    ``secondary_joint_velocity`` is a five-vector ``z``.

    MR: §6.3 ((I - GJ) term, no equation number)
    Lecture: L03 (exercise 3.3, slide 3.5)
    Means: pseudo-inverse and null-space projection
    (so101_classical_control.kinematics.inverse.null_space_projector)
    """
    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    secondary = _as_joint_velocity(secondary_joint_velocity)
    projector = null_space_projector(space_jacobian(angles), damping)
    return np.asarray(projector @ secondary, dtype=_DTYPE)


def tell_whether_planar_target_is_reachable(point_xy, l1: float, l2: float) -> bool:
    """Return whether a planar 2R tip can reach ``point_xy`` with links ``l1``, ``l2``.

    The reachable set is the annulus ``|l1 - l2| <= r <= l1 + l2``; points on a
    boundary count as reachable.

    MR: §6.1 (annular workspace, no equation number)
    Lecture: L03 (exercise 3.1)
    Means: so101_classical_control.kinematics.planar.planar_reachable
    """
    return planar_reachable(point_xy, l1, l2)


def report_manipulability(
    joint_angles,
    *,
    l1: float | None = None,
    l2: float | None = None,
    planar: bool | None = None,
) -> float:
    """Return the Yoshikawa manipulability of the arm at ``joint_angles``.

    For SO-101 the spatial Jacobian is ``6 x 5``, so ``det(J J^T)`` is always
    zero; the measure used here for a thin Jacobian (task dimension larger than
    the joint count) is the product of the singular values,
    ``sqrt(det(J^T J))``. For a planar 2R arm the ``2 x 2`` analytical positional
    Jacobian is used instead, so the measure is ``|l1 l2 sin(theta2)|`` and is
    zero at ``theta2 = 0`` and ``pi``.

    Pass ``planar=True`` (or ``l1`` and ``l2``) with a two-vector of joint angles
    for the planar arm; otherwise the five SO-101 arm angles are used.

    MR: §5.4 (mu3 = sqrt(det(J J^T)); no numbered MR equation)
    Lecture: L03 (exercise 2.2)
    Means: singular values of the Jacobian
    (so101_classical_control.kinematics.planar.planar_manipulability or
    so101_classical_control.kinematics.poe.space_jacobian)
    """
    use_planar = planar if planar is not None else (l1 is not None and l2 is not None)
    if use_planar:
        if l1 is None or l2 is None:
            raise ValueError("planar manipulability needs both l1 and l2")
        return planar_manipulability(_as_vector(joint_angles, 2, "joint_angles"), l1, l2)

    angles = np.asarray(arm_joint_angles(joint_angles), dtype=_DTYPE)
    jacobian = space_jacobian(angles)
    determinant = float(np.linalg.det(jacobian.T @ jacobian))
    return float(np.sqrt(max(0.0, determinant)))


# ---------------------------------------------------------------------------
# L02 frame-mapping and calibration goals
# ---------------------------------------------------------------------------


def express_camera_point_in_base_frame(T_base_cam, point_in_camera) -> np.ndarray:
    """Map a point given in the camera frame into the robot base frame.

    ``T_base_cam`` is the ``(4, 4)`` camera-to-base transform and
    ``point_in_camera`` a ``(3,)`` point; the result is ``R p + t``.

    MR: §3.3.1, Eq. (3.65)
    Lecture: L02 (frame mapping)
    Means: so101_classical_control.kinematics.transform_point
    """
    return transform_point(T_base_cam, point_in_camera)


def locate_top_camera_in_base_frame(base_marker, marker_in_camera) -> np.ndarray:
    """Return ``T_base_cam`` for the desk-fixed top camera from one closed loop.

    ``base_marker`` is ``T_base_marker`` of ``ee_marker`` and ``marker_in_camera``
    is the same marker seen in the camera; the result is
    ``T_base_marker @ inv(T_cam_marker)``.

    MR: §3.3.1, Eq. (3.64); cf. TransInv. Closed-loop composition is
    Eqs. (3.66)-(3.67).
    Lecture: L02 (closed-loop transform equation)
    Means: so101_classical_control.calibration.camera_to_base
    """
    return camera_to_base(base_marker, marker_in_camera)


def calibrate_wrist_camera_to_gripper(hand_poses, camera_poses) -> np.ndarray:
    """Return the fixed wrist-camera-to-gripper transform from relative motions.

    ``hand_poses`` and ``camera_poses`` are the ``gripperframe`` and ``wrist_cam``
    poses of the same arm configurations in the base frame. The returned ``X``
    satisfies ``A X = X B`` for every motion pair.

    MR: Ch. 3 application (geometry R_A R_X = R_X R_B); no numbered MR equation
    Lecture: L02 lecture notes section 3; exercise sections 10.2-10.3
    Means: so101_classical_control.calibration.build_motion_pairs and
    so101_classical_control.calibration.solve_ax_xb
    """
    a_list, b_list = build_motion_pairs(hand_poses, camera_poses)
    return solve_ax_xb(a_list, b_list)


def express_mujoco_camera_pose_in_opencv(transform) -> np.ndarray:
    """Return a MuJoCo camera pose converted to the OpenCV/optical convention.

    The rotation is right-multiplied by ``diag(1, -1, -1)``; the translation is
    unchanged.

    MR: §3.3.1, Eq. (3.67) (body-frame composition with a fixed rotation)
    Lecture: L01/L02 do not fix the camera-axis convention; this wrapper only
    applies the existing conversion.
    Means: so101_classical_control.kinematics.mujoco_camera_to_opencv
    """
    return mujoco_camera_to_opencv(transform)


# ---------------------------------------------------------------------------
# Applied1 sponge-to-paper task goals
# ---------------------------------------------------------------------------


def locate_sponge_in_base_frame(
    point_in_camera,
    *,
    T_base_cam=None,
    base_marker=None,
    marker_in_camera=None,
) -> np.ndarray:
    """Map a sponge point given in the camera frame into the robot base frame.

    The camera-frame point is an input; this is not pixel detection. Supply
    either ``T_base_cam`` directly, or ``base_marker`` and ``marker_in_camera``
    so the transform comes from :func:`locate_top_camera_in_base_frame`.

    MR: §3.3.1, Eq. (3.65) (same point-mapping as express_camera_point_in_base_frame)
    Lecture: L02 (frame mapping)
    Means: so101_classical_control.goals.express_camera_point_in_base_frame
    """
    if T_base_cam is not None:
        transform = _as_pose(T_base_cam, "T_base_cam")
    else:
        if base_marker is None or marker_in_camera is None:
            raise ValueError("provide T_base_cam, or both base_marker and marker_in_camera")
        transform = locate_top_camera_in_base_frame(base_marker, marker_in_camera)
    return express_camera_point_in_base_frame(transform, point_in_camera)


def plan_grasp_pose(
    sponge_pose_in_base,
    *,
    clearance: float = GRASP_CLEARANCE,
    yaw: float = 0.0,
) -> GraspPlan:
    """Plan a top-down gripperframe grasp pose from a sponge pose in the base frame.

    The tool finger axis (:data:`GRASP_FINGER_AXIS`, tool ``+X``) points along
    world ``-Z``, so the fingers descend onto the sponge from above; the jaws
    open along tool ``+Z``, which is horizontal and turns with ``yaw`` about
    world ``+Z``. The grasp pose puts the jaw midpoint
    (:data:`GRASP_PINCH_OFFSET`) on the sponge position; the pre-grasp pose is
    lifted ``clearance`` straight up. This is a geometric plan, not a motion, and
    it does not check reachability: on the vendored model the reachable
    orientation at the sponge height is this one, but the lifted pre-grasp pose
    of the same orientation is not necessarily reachable.

    MR: Ch. 4 / §4.1 (pose in the base frame; no numbered MR equation for a
    grasp heuristic)
    Lecture: Applied1 sponge-to-paper task, composed with L03 FK/IK
    Means: so101_classical_control.kinematics helpers via this module
    """
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"clearance must be finite and non-negative, got {clearance!r}")
    if not np.isfinite(yaw):
        raise ValueError(f"yaw must be finite, got {yaw!r}")
    sponge = _as_pose(sponge_pose_in_base, "sponge_pose_in_base")
    grasp_pose = _grasp_pose(sponge[:3, 3].copy(), yaw)
    return GraspPlan(
        pose=grasp_pose,
        pregrasp_pose=_lifted(grasp_pose, clearance),
        approach_direction=np.array([0.0, 0.0, 1.0], dtype=_DTYPE),
    )


def close_gripper_on_object() -> float:
    """Return the gripper joint command in radians that closes the jaws.

    About ``0 rad`` is closed on this hinge, and positive angles open the moving
    jaw; the value is conservative and inside ``[-0.174533, 1.7453292]``.

    MR: no numbered MR equation (gripper DoF is outside the five-joint arm chain)
    Lecture: Applied1 pick-and-place
    Means: backend joint command values
    """
    return float(GRIPPER_CLOSED_RAD)


def release_object() -> float:
    """Return the gripper joint command in radians that opens the jaws.

    Positive angles open the moving jaw; the value is conservative and inside
    ``[-0.174533, 1.7453292]``.

    MR: no numbered MR equation (gripper DoF is outside the five-joint arm chain)
    Lecture: Applied1 pick-and-place
    Means: backend joint command values
    """
    return float(GRIPPER_OPEN_RAD)


def carry_to_paper(
    paper_pose_in_base,
    *,
    sponge_pose_in_base=None,
    q0=None,
    clearance: float = 0.0,
    yaw: float = 0.0,
    joint_limits: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray | UnreachableTarget:
    """Return joint angles that carry the sponge to a top-down place pose at the paper.

    The place pose is derived from ``paper_pose_in_base`` with the same top-down
    convention as :func:`plan_grasp_pose`, lifted by ``clearance``. The solve
    goes through :func:`reach_target_pose`, so an unreachable place pose returns
    ``UnreachableTarget`` and joint limits are never silently clipped.
    ``sponge_pose_in_base`` is accepted so the caller can document which object is
    being carried; it does not change the place pose.

    MR: §6.2, Eq. (6.6) (goal directed through reach_target_pose; no separate
    carry equation)
    Lecture: L03 (slide 3.5, exercises 3.1-3.3) composed with the Applied1 task
    Means: so101_classical_control.goals.reach_target_pose
    """
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"clearance must be finite and non-negative, got {clearance!r}")
    paper = _as_pose(paper_pose_in_base, "paper_pose_in_base")
    if sponge_pose_in_base is not None:
        _as_pose(sponge_pose_in_base, "sponge_pose_in_base")
    place = _lifted(_grasp_pose(paper[:3, 3].copy(), yaw), clearance)
    return reach_target_pose(place, q0=q0, planar=False, joint_limits=joint_limits)
