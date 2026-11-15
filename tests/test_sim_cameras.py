"""Tests for the MuJoCo cameras, the pose getters, and the camera-frame helper.

Tolerances
----------
Rigid-transform and forward-kinematics comparisons use **1e-6 m** for
translation (``atol`` on the ``(3,)`` position, ``rtol=0``) and **1e-6 rad** for
rotation, where the rotation error is the angle of ``R1.T @ R2`` computed as
``arccos(clip((trace - 1) / 2, -1, 1))``.  This matches ``tests/test_dh_fk.py``.

The geometric visibility checks do not use image processing.  A point counts as
visible when it is in front of the camera (positive depth along the MuJoCo look
axis ``-Z``) and inside **90% of the half field of view** in both image axes,
i.e. ``|x| <= 0.9 * tan(hfov/2) * depth`` and ``|y| <= 0.9 * tan(fovy/2) *
depth``.  fovy is read from the compiled model, and the image aspect ratio is
the default 320x240.  Nothing here models occlusion, so "visible" means "inside
the frustum", which is the property the later calibration steps need.

Documented configuration
------------------------
The visibility checks pose the arm at
``VISIBILITY_POSE = (shoulder_pan=0.0, shoulder_lift=-0.1, elbow_flex=0.0,
wrist_flex=1.4, wrist_roll=0.0, gripper=0.0)`` rad, i.e. the tool pitched down
over the workspace with the jaws about 10 cm above the desk.  In that
configuration ``desk_marker`` is inside ``wrist_cam``'s frustum and ``ee_marker``
is inside ``top_cam``'s frustum.  The static-scene check sweeps several
configurations, including that one.

The pose getters are driven through the public API only (``command_joint_targets``
plus ``step``); the exact reference pose is ``forward_kinematics`` of the joint
positions read back afterwards, so servo settling error does not enter the
tolerance.

One tolerance is looser for a documented physical reason: while the ``gripper``
joint moves, the arm position servos give up a few ``1e-6`` rad holding the arm
against the reaction torque of the moving jaw, which shifts every frame mounted
upstream of that jaw.  The rigidity checks are therefore stated relative to
``gripperframe`` (exact to 1e-6) and only bound the residual base-frame drift by
``GRIPPER_REACTION_TOL``.

No OpenCV, no intrinsic matrix, and no marker detection are used here.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from so101_classical_control.backend import CAMERA_NAMES, JOINT_NAMES
from so101_classical_control.kinematics import (
    forward_kinematics,
    inverse_se3,
    mujoco_camera_to_opencv,
    pose,
    rotation_of,
    translation_of,
)
from so101_classical_control.sim import SimBackend
from so101_classical_control.sim.scene import scene_xml_path

SEED = 20261114
FK_SAMPLES = 30
TRANSLATION_TOL = 1e-6  # metres
ROTATION_TOL = 1e-6  # radians
#: Fraction of the half field of view a marker must stay inside.
FRUSTUM_MARGIN = 0.9
#: Joint-limit margin used to sample configurations inside the joint stops.
LIMIT_MARGIN = 1e-2  # radians
#: Bound on the arm drift caused by the moving jaw's reaction torque while the
#: ``gripper`` joint moves.  The observed drift is a few ``1e-6`` m / rad; a frame
#: attached to ``moving_jaw_so101_v1`` would instead move by centimetres.
GRIPPER_REACTION_TOL = 1e-4  # metres / radians
#: Default rendering size, which also fixes the image aspect ratio.
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240

SITE_NAMES = ("baseframe", "gripperframe", "desk_marker", "ee_marker")

#: Documented arm configuration used by the visibility checks.  Joint values in
#: :data:`JOINT_NAMES` order, radians.
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
    """The compiled scene, used for joint ranges and camera field of view."""
    return mujoco.MjModel.from_xml_path(str(scene_xml_path()))


def _joint_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, name
    low, high = model.jnt_range[joint_id]
    return float(low), float(high)


def _fovy_degrees(model: mujoco.MjModel, camera: str) -> float:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    assert camera_id != -1, camera
    # MuJoCo stores the vertical field of view in degrees.
    return float(model.cam_fovy[camera_id])


def _pose_arm(backend: SimBackend, targets: dict[str, float]) -> dict[str, float]:
    """Command ``targets``, settle, and return the joint positions actually read."""
    backend.command_joint_targets(targets)
    backend.step(1.0)
    return backend.read_joint_positions()


def _rotation_error(r_left: np.ndarray, r_right: np.ndarray) -> float:
    """Return the angle in radians of ``r_left.T @ r_right``."""
    cosine = np.clip((np.trace(r_left.T @ r_right) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _named_targets(model: mujoco.MjModel, case: str) -> dict[str, float]:
    ranges = {name: _joint_range(model, name) for name in JOINT_NAMES}
    if case == "zeros":
        return dict.fromkeys(JOINT_NAMES, 0.0)
    if case == "mid_range":
        return {name: 0.5 * (low + high) for name, (low, high) in ranges.items()}
    if case == "near_lower":
        return {name: low + LIMIT_MARGIN for name, (low, _) in ranges.items()}
    if case == "near_upper":
        return {name: high - LIMIT_MARGIN for name, (_, high) in ranges.items()}
    raise AssertionError(case)


def _random_targets(model: mujoco.MjModel, rng: np.random.Generator) -> dict[str, float]:
    targets = {}
    for name in JOINT_NAMES:
        low, high = _joint_range(model, name)
        targets[name] = float(rng.uniform(low + LIMIT_MARGIN, high - LIMIT_MARGIN))
    return targets


def _frustum_coordinates(
    camera_pose: np.ndarray, point: np.ndarray, fovy_deg: float, aspect: float
) -> tuple[float, float, float]:
    """Return ``(depth, x_norm, y_norm)`` of ``point`` in the camera image.

    ``depth`` is measured along the MuJoCo look axis (``-Z`` of the camera
    frame).  ``x_norm`` and ``y_norm`` are the in-plane offsets divided by the
    half-width and half-height of the frustum at that depth, so ``|x_norm| <= 1``
    and ``|y_norm| <= 1`` mean "inside the image".
    """
    point_camera = rotation_of(camera_pose).T @ (
        np.asarray(point, dtype=np.float64) - translation_of(camera_pose)
    )
    depth = float(-point_camera[2])
    tan_y = math.tan(math.radians(fovy_deg) / 2.0)
    tan_x = tan_y * aspect
    x_norm = float(point_camera[0] / (tan_x * depth))
    y_norm = float(point_camera[1] / (tan_y * depth))
    return depth, x_norm, y_norm


def _assert_in_frustum(
    camera_pose: np.ndarray, point: np.ndarray, fovy_deg: float, label: str
) -> None:
    aspect = DEFAULT_WIDTH / DEFAULT_HEIGHT
    depth, x_norm, y_norm = _frustum_coordinates(camera_pose, point, fovy_deg, aspect)
    assert depth > 0.0, f"{label} is behind the camera (depth {depth:.3f} m)"
    assert abs(x_norm) <= FRUSTUM_MARGIN, f"{label} is outside the image in x ({x_norm:.3f})"
    assert abs(y_norm) <= FRUSTUM_MARGIN, f"{label} is outside the image in y ({y_norm:.3f})"


def _assert_se3(transform: np.ndarray, label: str) -> None:
    assert isinstance(transform, np.ndarray), label
    assert transform.shape == (4, 4), label
    assert transform.dtype == np.float64, label
    np.testing.assert_array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12), label


# ---------------------------------------------------------------------------
# The scene provides both cameras and both marker sites
# ---------------------------------------------------------------------------


def test_camera_names_match_the_scene(model: mujoco.MjModel) -> None:
    scene_cameras = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(model.ncam)
    }
    assert set(CAMERA_NAMES) == scene_cameras
    assert CAMERA_NAMES == ("top_cam", "wrist_cam")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera", CAMERA_NAMES)
def test_camera_renders_at_the_default_size(camera: str) -> None:
    with SimBackend() as backend:
        # Rendering before the first step exercises the initial forward pass.
        frame = backend.get_camera_image(camera)
        backend.step(0.1)
        post_step_frame = backend.get_camera_image(camera)

    for image in (frame, post_step_frame):
        assert image.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
        assert image.dtype == np.uint8
        assert image.size > 0
        # A flat frame would collapse to a single value.
        assert np.unique(image).size > 1


def test_unknown_camera_still_raises() -> None:
    with SimBackend() as backend:
        with pytest.raises(KeyError, match="unknown camera"):
            backend.get_camera_image("does_not_exist")
        with pytest.raises(KeyError, match="unknown camera"):
            backend.get_camera_pose("does_not_exist")


# ---------------------------------------------------------------------------
# Pose getters: contract, errors, lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera", CAMERA_NAMES)
def test_camera_pose_is_a_valid_se3(camera: str) -> None:
    with SimBackend() as backend:
        _assert_se3(backend.get_camera_pose(camera), camera)


@pytest.mark.parametrize("site", SITE_NAMES)
def test_site_pose_is_a_valid_se3(site: str) -> None:
    with SimBackend() as backend:
        _assert_se3(backend.get_site_pose(site), site)


def test_baseframe_pose_is_the_identity() -> None:
    with SimBackend() as backend:
        np.testing.assert_allclose(
            backend.get_site_pose("baseframe"), np.eye(4), atol=1e-12, rtol=0.0
        )


def test_unknown_site_raises_key_error() -> None:
    with SimBackend() as backend:
        with pytest.raises(KeyError, match="unknown site"):
            backend.get_site_pose("does_not_exist")


def test_pose_getters_fail_after_close() -> None:
    backend = SimBackend()
    backend.close()

    with pytest.raises(RuntimeError, match="closed"):
        backend.get_camera_pose("wrist_cam")
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_site_pose("gripperframe")


# ---------------------------------------------------------------------------
# gripperframe agrees with the DH forward kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["zeros", "mid_range", "near_lower", "near_upper"])
def test_site_pose_matches_forward_kinematics_named_cases(
    model: mujoco.MjModel, case: str
) -> None:
    with SimBackend() as backend:
        positions = _pose_arm(backend, _named_targets(model, case))
        site_pose = backend.get_site_pose("gripperframe")

    expected = forward_kinematics(positions)
    np.testing.assert_allclose(
        site_pose[:3, 3], expected[:3, 3], atol=TRANSLATION_TOL, rtol=0.0, err_msg=case
    )
    error = _rotation_error(site_pose[:3, :3], expected[:3, :3])
    assert error <= ROTATION_TOL, f"{case}: rotation error {error:.3e} rad"


def test_site_pose_matches_forward_kinematics_random_configurations(
    model: mujoco.MjModel,
) -> None:
    rng = np.random.default_rng(SEED)
    with SimBackend() as backend:
        for index in range(FK_SAMPLES):
            positions = _pose_arm(backend, _random_targets(model, rng))
            site_pose = backend.get_site_pose("gripperframe")
            expected = forward_kinematics(positions)

            np.testing.assert_allclose(
                site_pose[:3, 3],
                expected[:3, 3],
                atol=TRANSLATION_TOL,
                rtol=0.0,
                err_msg=f"random sample {index}",
            )
            error = _rotation_error(site_pose[:3, :3], expected[:3, :3])
            assert error <= ROTATION_TOL, f"random sample {index}: {error:.3e} rad"


# ---------------------------------------------------------------------------
# Which frames move with what
# ---------------------------------------------------------------------------


def test_ee_marker_and_wrist_cam_are_fixed_to_the_gripper_body() -> None:
    """``ee_marker`` and ``wrist_cam`` must not move with the ``gripper`` joint.

    The exact statement checked here is rigidity with respect to ``gripperframe``:
    both frames hang off the fixed ``gripper`` body, so ``inv(T_gf) @ T`` must be
    unchanged while the ``gripper`` joint sweeps.

    Their base-frame poses do change slightly, because the arm position servos
    give up a few ``1e-6`` rad holding the arm against the reaction torque of the
    moving jaw.  That drift is bounded by :data:`GRIPPER_REACTION_TOL`; a frame
    attached to ``moving_jaw_so101_v1`` instead would move by centimetres.
    """
    with SimBackend() as backend:
        arm_only = {
            name: value for name, value in VISIBILITY_POSE.items() if name != "gripper"
        }
        backend.command_joint_targets(arm_only)
        backend.step(1.0)

        reference_positions = backend.read_joint_positions()
        reference_gf = backend.get_site_pose("gripperframe")
        reference_ee = backend.get_site_pose("ee_marker")
        reference_cam = backend.get_camera_pose("wrist_cam")
        reference_relative_ee = inverse_se3(reference_gf) @ reference_ee
        reference_relative_cam = inverse_se3(reference_gf) @ reference_cam

        for gripper in (-0.17453, 0.0, 0.5, 1.0, 1.74533):
            backend.command_joint_targets({"gripper": gripper})
            backend.step(1.0)
            positions = backend.read_joint_positions()
            assert positions["gripper"] == pytest.approx(gripper, abs=0.05)

            # Rigid with respect to the fixed gripper body: exact to 1e-6.
            gripperframe = backend.get_site_pose("gripperframe")
            np.testing.assert_allclose(
                inverse_se3(gripperframe) @ backend.get_site_pose("ee_marker"),
                reference_relative_ee,
                atol=TRANSLATION_TOL,
                rtol=0.0,
                err_msg=f"ee_marker moved relative to gripperframe at gripper={gripper}",
            )
            np.testing.assert_allclose(
                inverse_se3(gripperframe) @ backend.get_camera_pose("wrist_cam"),
                reference_relative_cam,
                atol=TRANSLATION_TOL,
                rtol=0.0,
                err_msg=f"wrist_cam moved relative to gripperframe at gripper={gripper}",
            )

            # Only the documented servo drift of the arm joints is left.
            arm_shift = max(
                abs(positions[name] - reference_positions[name]) for name in arm_only
            )
            assert arm_shift <= GRIPPER_REACTION_TOL, (gripper, arm_shift)
            base_frame_shift = max(
                float(np.abs(backend.get_site_pose("ee_marker") - reference_ee).max()),
                float(np.abs(backend.get_camera_pose("wrist_cam") - reference_cam).max()),
            )
            assert base_frame_shift <= GRIPPER_REACTION_TOL, (gripper, base_frame_shift)


def test_desk_marker_and_top_cam_are_static() -> None:
    with SimBackend() as backend:
        backend.step(1.0)
        reference_marker = backend.get_site_pose("desk_marker")
        reference_cam = backend.get_camera_pose("top_cam")

        for targets in (
            dict.fromkeys(JOINT_NAMES, 0.0),
            VISIBILITY_POSE,
            {
                "shoulder_pan": 0.6,
                "shoulder_lift": 0.4,
                "elbow_flex": -0.5,
                "wrist_flex": 0.2,
                "wrist_roll": -1.0,
                "gripper": 0.3,
            },
        ):
            _pose_arm(backend, targets)
            np.testing.assert_allclose(
                backend.get_site_pose("desk_marker"),
                reference_marker,
                atol=TRANSLATION_TOL,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                backend.get_camera_pose("top_cam"), reference_cam, atol=TRANSLATION_TOL, rtol=0.0
            )


# ---------------------------------------------------------------------------
# Eye-in-hand rigidity: T_gripperframe_to_wrist_cam is constant
# ---------------------------------------------------------------------------


def test_eye_in_hand_transform_is_constant(model: mujoco.MjModel) -> None:
    rng = np.random.default_rng(SEED)
    mid_range = {
        name: sum(_joint_range(model, name)) / 2.0 for name in JOINT_NAMES[:5]
    }
    configurations: list[dict[str, float]] = [
        dict.fromkeys(JOINT_NAMES, 0.0),
        VISIBILITY_POSE,
        {**mid_range, "gripper": 0.0},
    ]
    configurations.extend(_random_targets(model, rng) for _ in range(5))

    with SimBackend() as backend:
        reference: np.ndarray | None = None
        for index, targets in enumerate(configurations):
            _pose_arm(backend, targets)
            transform = inverse_se3(backend.get_site_pose("gripperframe")) @ (
                backend.get_camera_pose("wrist_cam")
            )
            if reference is None:
                reference = transform
                continue
            np.testing.assert_allclose(
                transform[:3, 3],
                reference[:3, 3],
                atol=TRANSLATION_TOL,
                rtol=0.0,
                err_msg=f"eye-in-hand translation drifted at configuration {index}",
            )
            error = _rotation_error(transform[:3, :3], reference[:3, :3])
            assert error <= ROTATION_TOL, f"configuration {index}: {error:.3e} rad"


# ---------------------------------------------------------------------------
# Geometric visibility in the documented configuration
# ---------------------------------------------------------------------------


def test_desk_marker_is_visible_from_wrist_cam(model: mujoco.MjModel) -> None:
    with SimBackend() as backend:
        positions = _pose_arm(backend, VISIBILITY_POSE)
        for name, value in VISIBILITY_POSE.items():
            assert positions[name] == pytest.approx(value, abs=0.01), name
        camera_pose = backend.get_camera_pose("wrist_cam")
        marker_pose = backend.get_site_pose("desk_marker")

    _assert_in_frustum(
        camera_pose, translation_of(marker_pose), _fovy_degrees(model, "wrist_cam"), "desk_marker"
    )


def test_ee_marker_is_visible_from_top_cam(model: mujoco.MjModel) -> None:
    with SimBackend() as backend:
        positions = _pose_arm(backend, VISIBILITY_POSE)
        for name, value in VISIBILITY_POSE.items():
            assert positions[name] == pytest.approx(value, abs=0.01), name
        camera_pose = backend.get_camera_pose("top_cam")
        marker_pose = backend.get_site_pose("ee_marker")

    _assert_in_frustum(
        camera_pose, translation_of(marker_pose), _fovy_degrees(model, "top_cam"), "ee_marker"
    )


# ---------------------------------------------------------------------------
# mujoco_camera_to_opencv
# ---------------------------------------------------------------------------


def test_mujoco_camera_to_opencv_known_case() -> None:
    # MuJoCo camera frame: +X right, +Y up, looking along -Z.
    transform = pose(np.eye(3), np.array([0.1, 0.2, 0.3]))
    converted = mujoco_camera_to_opencv(transform)

    rotation = rotation_of(converted)
    # +X right is shared by both conventions.
    np.testing.assert_allclose(rotation[:, 0], [1.0, 0.0, 0.0], atol=0.0)
    # +Y up (MuJoCo) becomes -Y (OpenCV): +Y in OpenCV is down.
    np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=0.0)
    # The look axis -Z (MuJoCo) becomes +Z (OpenCV).
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=0.0)
    # The translation is unchanged.
    np.testing.assert_allclose(translation_of(converted), translation_of(transform), atol=0.0)

    # A MuJoCo look direction of -Z maps to an OpenCV look direction of +Z.
    mj_look = rotation_of(transform) @ np.array([0.0, 0.0, -1.0])
    cv_look = rotation_of(converted) @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(mj_look, cv_look, atol=1e-15)


def test_mujoco_camera_to_opencv_round_trips() -> None:
    transform = pose(
        np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        np.array([-0.2, 0.05, 0.4]),
    )
    twice = mujoco_camera_to_opencv(mujoco_camera_to_opencv(transform))
    np.testing.assert_allclose(twice, transform, atol=1e-15, rtol=0.0)


def test_mujoco_camera_to_opencv_converts_backend_poses() -> None:
    with SimBackend() as backend:
        for camera in CAMERA_NAMES:
            mj_pose = backend.get_camera_pose(camera)
            cv_pose = mujoco_camera_to_opencv(mj_pose)
            _assert_se3(cv_pose, camera)
            # The optical axis (+Z in OpenCV) is the MuJoCo look direction (-Z).
            np.testing.assert_allclose(
                rotation_of(cv_pose) @ np.array([0.0, 0.0, 1.0]),
                rotation_of(mj_pose) @ np.array([0.0, 0.0, -1.0]),
                atol=1e-15,
            )


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(3),
        np.zeros((4, 4)),
        np.eye(5),
        np.eye(4).tolist()[:3],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
    ],
)
def test_mujoco_camera_to_opencv_rejects_invalid_shape(transform) -> None:
    with pytest.raises(ValueError):
        mujoco_camera_to_opencv(transform)
