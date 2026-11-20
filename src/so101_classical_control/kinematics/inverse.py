"""Damped least-squares inverse kinematics and null-space projection.

This module provides the numerical means behind the goal-level
:func:`so101_classical_control.goals.reach_target_pose` and
:func:`so101_classical_control.goals.keep_gripper_fixed_to_first_order`. It
builds on :mod:`~so101_classical_control.kinematics.poe` (the ``FKinSpace`` /
``JacobianSpace`` wrappers) and on ``modern_robotics``; it does not re-derive a
solver that already exists.

The iterative solver is resolved-rate control with a damped pseudo-inverse and
backtracking line search. A candidate step is only accepted when it does not
increase the position error, so the reported solution is never a step that made
things worse (P11). A run is a success only when the forward kinematics of the
returned angles meets the stop condition, never on the basis of a small
increment alone (P14).

Stopping tolerances
-------------------
:data:`IK_TRANSLATION_TOL` and :data:`IK_ROTATION_TOL` transcribe the lecture /
MR stopping values (``ev = 1e-4`` m and ``eω = 1e-3`` rad in ``IKinSpace``).
They are **provisional constants pending Yohei approval** (L03 plan Page,
section "L03 資料を踏まえた性質リスト（承認待ち案）").
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from modern_robotics import MatrixLog6, se3ToVec

from .poe import fk_in_space, space_jacobian

__all__ = [
    "IK_DAMPING",
    "IK_MAX_ITERATIONS",
    "IK_MAX_STEP",
    "IK_RESULT_SUCCESS_TOL",
    "IK_ROTATION_TOL",
    "IK_TRANSLATION_TOL",
    "IKResult",
    "damped_pseudo_inverse",
    "null_space_projector",
    "pose_error",
    "position_jacobian",
    "solve_pose_ik",
    "solve_position_ik",
]

_DTYPE = np.float64

#: Damping ``lambda`` of the damped pseudo-inverse. Small enough to solve the
#: well-conditioned directions, large enough to stay finite at a singularity.
IK_DAMPING: Final[float] = 1e-3
#: Iteration cap of the numerical solver.
IK_MAX_ITERATIONS: Final[int] = 200
#: Per-joint step cap in radians, applied to the norm of the joint update.
IK_MAX_STEP: Final[float] = 0.2
#: Translation stop ``ev`` in metres. **Provisional, pending Yohei approval.**
IK_TRANSLATION_TOL: Final[float] = 1e-4
#: Rotation stop ``eomega`` in radians. **Provisional, pending Yohei approval.**
IK_ROTATION_TOL: Final[float] = 1e-3
#: Relative slack used when comparing two position errors in the line search.
IK_RESULT_SUCCESS_TOL: Final[float] = 1e-12
#: The iteration continues while the residual exceeds this fraction of the
#: documented stop, so an accepted solution has margin rather than sitting on
#: the boundary of the stop condition.
_INNER_TOL_FRACTION: Final[float] = 0.1

#: Singular values below this fraction of the largest are treated as zero when
#: ``damping == 0``, so the returned map is the Moore-Penrose pseudo-inverse.
_RANK_TOL: Final[float] = 1e-12


@dataclass(frozen=True)
class IKResult:
    """Outcome of a numerical inverse-kinematics solve.

    Attributes
    ----------
    joint_angles
        The ``(5,)`` float64 solution, or the seed when nothing was accepted.
    success
        ``True`` only when the forward kinematics of ``joint_angles`` meets the
        stop condition.
    iterations
        Number of accepted iterations.
    position_error
        Euclidean position residual of ``joint_angles`` in metres.
    rotation_error
        Rotation residual in radians, or ``None`` for a position-only solve.
    reason
        Short English status; ``"converged"`` on success.
    """

    joint_angles: np.ndarray
    success: bool
    iterations: int
    position_error: float
    rotation_error: float | None
    reason: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_arm_angles(value: object, name: str) -> np.ndarray:
    """Return ``value`` as a finite ``(5,)`` float64 arm-angle array."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float64 array") from error
    if array.shape != (5,):
        raise ValueError(f"{name} must have shape (5,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_jacobian(value: object) -> np.ndarray:
    """Return ``value`` as a finite 2-D float64 Jacobian."""
    try:
        array = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError("jacobian must be convertible to a float64 array") from error
    if array.ndim != 2:
        raise ValueError(f"jacobian must be two-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("jacobian must contain only finite values")
    return array


def _as_nonnegative(value: object, name: str) -> float:
    """Return ``value`` as a finite non-negative float."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return scalar


# ---------------------------------------------------------------------------
# Jacobians and pseudo-inverses
# ---------------------------------------------------------------------------


def position_jacobian(joint_angles) -> np.ndarray:
    """Return the ``(3, 5)`` Jacobian ``d(position) / d(joint_angles)``.

    The spatial Jacobian column ``i`` is the twist ``(omega_i, v_i)`` that moves
    the end-effector frame; the velocity of the frame origin is
    ``omega_i x p + v_i`` with ``p`` the end-effector position (MR's twist
    convention). This relation makes the position Jacobian a function of the
    space Jacobian rather than a separate derivation.

    Raises
    ------
    ValueError
        If ``joint_angles`` is malformed or non-finite.
    """
    angles = np.asarray(joint_angles, dtype=_DTYPE)
    jacobian = space_jacobian(angles)
    position = fk_in_space(angles)[:3, 3]
    return jacobian[3:, :] + np.cross(jacobian[:3, :].T, position).T


def damped_pseudo_inverse(jacobian, damping: float = 0.0) -> np.ndarray:
    """Return the damped Moore-Penrose pseudo-inverse of ``jacobian``.

    With the SVD ``J = U diag(s) V^T`` the returned map is
    ``V diag(s / (s^2 + damping^2)) U^T``. ``damping == 0`` reduces to the exact
    Moore-Penrose pseudo-inverse (singular values at or below
    :data:`_RANK_TOL` times the largest are dropped), which is the limit that
    makes the null-space projector an exact projector.

    Raises
    ------
    ValueError
        If ``jacobian`` is not a finite 2-D array or ``damping`` is negative.
    """
    array = _as_jacobian(jacobian)
    damping = _as_nonnegative(damping, "damping")
    u_matrix, singular_values, vt_matrix = np.linalg.svd(array, full_matrices=False)
    if damping == 0.0:
        largest = float(singular_values[0]) if singular_values.size else 0.0
        cutoff = _RANK_TOL * max(largest, 1.0)
        inverse_values = np.array(
            [1.0 / value if value > cutoff else 0.0 for value in singular_values],
            dtype=_DTYPE,
        )
    else:
        inverse_values = singular_values / (singular_values**2 + damping**2)
    return (vt_matrix.T * inverse_values) @ u_matrix.T


def null_space_projector(jacobian, damping: float = 0.0) -> np.ndarray:
    """Return ``I - J^+ J``, the projector onto the null space of ``jacobian``.

    ``J^+`` is :func:`damped_pseudo_inverse`; with ``damping == 0`` and a full
    row rank this is an exact orthogonal projector, so ``J (I - J^+ J) z`` is
    zero to rounding for every ``z`` (P12).

    Raises
    ------
    ValueError
        If ``jacobian`` is not a finite 2-D array or ``damping`` is negative.
    """
    array = _as_jacobian(jacobian)
    identity = np.eye(array.shape[1], dtype=_DTYPE)
    return identity - damped_pseudo_inverse(array, damping) @ array


# ---------------------------------------------------------------------------
# Pose error
# ---------------------------------------------------------------------------


def _rotation_angle(rotation: np.ndarray) -> float:
    """Return the rotation angle of ``rotation`` in ``[0, pi]``."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Return the nearest proper rotation to ``rotation`` (SVD projection)."""
    u_matrix, _, vt_matrix = np.linalg.svd(rotation)
    projected = u_matrix @ vt_matrix
    if np.linalg.det(projected) < 0.0:
        u_matrix[:, -1] *= -1.0
        projected = u_matrix @ vt_matrix
    return projected


def _space_error_twist(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the space-frame error twist ``se3ToVec(Log6(target inv(current)))``.

    This is the same error ``modern_robotics.IKinSpace`` forms (its body-frame
    log mapped to the space frame by an adjoint). The relative rotation is
    projected back onto SO(3) first, because a rotation built from a matrix
    product can have a trace a few ulp outside ``[-1, 3]`` and ``MatrixLog6``
    takes an ``arccos`` of it.
    """
    relative = target @ np.linalg.inv(current)
    relative = relative.copy()
    relative[:3, :3] = _project_to_so3(relative[:3, :3])
    return np.asarray(se3ToVec(MatrixLog6(relative)), dtype=_DTYPE)


def pose_error(target: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return ``(position_error_m, rotation_error_rad)`` between two SE(3) poses.

    The rotation error is the angle of ``R_target^T R_current``. Both inputs must
    be finite ``(4, 4)`` arrays with a proper rotation block.

    Raises
    ------
    ValueError
        If either input is not a valid ``(4, 4)`` homogeneous transform.
    """
    target_pose = np.asarray(target, dtype=_DTYPE)
    current_pose = np.asarray(current, dtype=_DTYPE)
    for name, transform in (("target", target_pose), ("current", current_pose)):
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
        if not np.all(np.isfinite(transform)):
            raise ValueError(f"{name} must contain only finite values")
        _rotation_angle(transform[:3, :3])
    position = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    rotation = _rotation_angle(target_pose[:3, :3].T @ current_pose[:3, :3])
    return position, rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limit_step(step: np.ndarray, max_step: float) -> np.ndarray:
    """Scale ``step`` so its infinity norm is at most ``max_step``."""
    largest = float(np.max(np.abs(step))) if step.size else 0.0
    if largest > max_step:
        return step * (max_step / largest)
    return step


def _within_limits(angles: np.ndarray, limits: tuple[tuple[float, float], ...]) -> bool:
    """Return whether ``angles`` lies inside the inclusive joint limits."""
    return all(
        low <= float(value) <= high
        for value, (low, high) in zip(angles, limits, strict=True)
    )


def _as_limits(limits) -> tuple[tuple[float, float], ...] | None:
    """Validate a per-joint ``(low, high)`` limit sequence of length five."""
    if limits is None:
        return None
    items = [tuple(float(bound) for bound in pair) for pair in limits]
    if len(items) != 5:
        raise ValueError(f"joint_limits must have five entries, got {len(items)}")
    for low, high in items:
        if not (math.isfinite(low) and math.isfinite(high)) or low > high:
            raise ValueError(f"invalid joint limit ({low}, {high})")
    return tuple(items)


# ---------------------------------------------------------------------------
# Numerical IK
# ---------------------------------------------------------------------------


def _backtrack(
    jacobian: np.ndarray,
    error_vector: np.ndarray,
    angles: np.ndarray,
    current_error: float,
    error_of,
    *,
    damping: float,
    max_step: float,
    limits: tuple[tuple[float, float], ...] | None,
) -> tuple[np.ndarray | None, float]:
    """Return ``(accepted_angles, accepted_error)`` or ``(None, current_error)``.

    The candidate direction is the damped pseudo-inverse applied to the error
    vector; every halving of the step is rejected if it leaves the joint limits
    or does not keep the position error non-increasing.
    """
    step = _limit_step(damped_pseudo_inverse(jacobian, damping) @ error_vector, max_step)
    if not np.all(np.isfinite(step)):
        return None, current_error
    scale = 1.0
    for _ in range(32):
        candidate = angles + scale * step
        if not np.all(np.isfinite(candidate)):
            scale *= 0.5
            continue
        if limits is not None and not _within_limits(candidate, limits):
            scale *= 0.5
            continue
        candidate_error = error_of(candidate)
        if candidate_error <= current_error * (1.0 + IK_RESULT_SUCCESS_TOL) + 1e-15:
            return candidate, candidate_error
        scale *= 0.5
    return None, current_error


def solve_position_ik(
    target_position: Sequence[float],
    seed: Sequence[float],
    *,
    damping: float = IK_DAMPING,
    max_iterations: int = IK_MAX_ITERATIONS,
    translation_tol: float = IK_TRANSLATION_TOL,
    max_step: float = IK_MAX_STEP,
    joint_limits: Sequence[tuple[float, float]] | None = None,
) -> IKResult:
    """Solve the five arm joints so ``gripperframe`` reaches ``target_position``.

    This is resolved-rate control on the ``(3, 5)`` position Jacobian with a
    damped pseudo-inverse, a per-joint step cap, and backtracking: a candidate
    step is only accepted when the position error does not increase (P11). The
    solve is a success only when the forward kinematics of the returned angles is
    within ``translation_tol`` (P14); otherwise ``success`` is ``False`` and the
    caller must not treat the angles as a solution.

    Raises
    ------
    ValueError
        If the target is not a finite 3-vector, the seed is not a finite
        5-vector, or the tolerances / limits are malformed.
    """
    target = np.asarray(target_position, dtype=_DTYPE)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_position must be a finite (3,) vector")
    angles = _as_arm_angles(seed, "seed")
    damping = _as_nonnegative(damping, "damping")
    translation_tol = _as_nonnegative(translation_tol, "translation_tol")
    max_step = _as_nonnegative(max_step, "max_step")
    if max_step == 0.0:
        raise ValueError("max_step must be positive")
    limits = _as_limits(joint_limits)
    if limits is not None and not _within_limits(angles, limits):
        raise ValueError("seed lies outside joint_limits")

    def error_of(candidate: np.ndarray) -> float:
        return float(np.linalg.norm(target - fk_in_space(candidate)[:3, 3]))

    current_error = error_of(angles)
    iterations = 0
    inner_tol = _INNER_TOL_FRACTION * translation_tol
    for _ in range(max_iterations):
        if current_error <= inner_tol:
            break
        jacobian = position_jacobian(angles)
        error_vector = target - fk_in_space(angles)[:3, 3]
        accepted, accepted_error = _backtrack(
            jacobian,
            error_vector,
            angles,
            current_error,
            error_of,
            damping=damping,
            max_step=max_step,
            limits=limits,
        )
        if accepted is None:
            break
        improvement = current_error - accepted_error
        angles, current_error = accepted, accepted_error
        iterations += 1
        if improvement <= 1e-15:
            break

    success = current_error <= translation_tol
    return IKResult(
        joint_angles=angles.copy(),
        success=success,
        iterations=iterations,
        position_error=current_error,
        rotation_error=None,
        reason="converged" if success else "no convergence",
    )


def solve_pose_ik(
    target_pose: np.ndarray,
    seed: Sequence[float],
    *,
    damping: float = IK_DAMPING,
    max_iterations: int = IK_MAX_ITERATIONS,
    translation_tol: float = IK_TRANSLATION_TOL,
    rotation_tol: float = IK_ROTATION_TOL,
    max_step: float = IK_MAX_STEP,
    joint_limits: Sequence[tuple[float, float]] | None = None,
) -> IKResult:
    """Solve the five arm joints toward a full ``(4, 4)`` ``gripperframe`` pose.

    The arm has five joints, so an arbitrary SE(3) target can be out of reach
    even when its position is inside the workspace. The solver is resolved-rate
    control on the ``(6, 5)`` space Jacobian with a damped pseudo-inverse and a
    backtracking line search that keeps the position error non-increasing (P11).
    Success requires the forward kinematics of the returned angles to meet both
    ``translation_tol`` and ``rotation_tol`` (P14).

    Raises
    ------
    ValueError
        If ``target_pose`` is not a finite ``(4, 4)`` transform, the seed is not
        a finite 5-vector, or the tolerances / limits are malformed.
    """
    target = np.asarray(target_pose, dtype=_DTYPE)
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        raise ValueError("target_pose must be a finite (4, 4) transform")
    angles = _as_arm_angles(seed, "seed")
    damping = _as_nonnegative(damping, "damping")
    translation_tol = _as_nonnegative(translation_tol, "translation_tol")
    rotation_tol = _as_nonnegative(rotation_tol, "rotation_tol")
    max_step = _as_nonnegative(max_step, "max_step")
    if max_step == 0.0:
        raise ValueError("max_step must be positive")
    limits = _as_limits(joint_limits)
    if limits is not None and not _within_limits(angles, limits):
        raise ValueError("seed lies outside joint_limits")

    def position_error_of(candidate: np.ndarray) -> float:
        return float(np.linalg.norm(target[:3, 3] - fk_in_space(candidate)[:3, 3]))

    current_error = position_error_of(angles)
    iterations = 0
    inner_translation = _INNER_TOL_FRACTION * translation_tol
    inner_rotation = _INNER_TOL_FRACTION * rotation_tol
    for _ in range(max_iterations):
        current = fk_in_space(angles)
        _, rotation_error = pose_error(target, current)
        if current_error <= inner_translation and rotation_error <= inner_rotation:
            break
        jacobian = space_jacobian(angles)
        error_twist = _space_error_twist(target, current)
        accepted, accepted_error = _backtrack(
            jacobian,
            error_twist,
            angles,
            current_error,
            position_error_of,
            damping=damping,
            max_step=max_step,
            limits=limits,
        )
        if accepted is None:
            break
        improvement = current_error - accepted_error
        angles, current_error = accepted, accepted_error
        iterations += 1
        if improvement <= 1e-15:
            break

    _, final_rotation_error = pose_error(target, fk_in_space(angles))
    success = current_error <= translation_tol and final_rotation_error <= rotation_tol
    return IKResult(
        joint_angles=angles.copy(),
        success=success,
        iterations=iterations,
        position_error=current_error,
        rotation_error=final_rotation_error,
        reason="converged" if success else "no convergence",
    )
