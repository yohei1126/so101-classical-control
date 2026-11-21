"""Reproducibility tests for the ``docs/crosscheck`` artifacts (Issue #33).

``docs/crosscheck`` holds a standalone SO-101 table scene plus the recorded
Robotics-plugin FK and camera poses it was checked against for Issue #24.
Issue #33 keeps that pair loadable and re-derivable from this repository
alone:

1. ``crosscheck_scene.xml`` must load from the repo root.  Its ``meshdir``
   points at the vendored menagerie STLs using the same relative-to-XML
   convention as ``sponge_paper_scene.xml``; the aloha-table OBJ/PNG assets,
   which are not vendored here, are dropped and only the table collision box
   remains.
2. The joint vectors recorded in the JSON ``q`` field (Issue #24) must
   reproduce the recorded ``fk.q0``..``fk.q4`` 4x4 poses: the pose of the
   ``robot/gripperframe`` site expressed in ``robot/baseframe`` after
   ``mj_forward``.
3. The recorded ``model.link_lengths_m`` and ``model.joint_limits_rad`` must
   still match the vendored ``so101.xml``.

Tolerances
----------
FK agreement uses ``1e-9`` on ``max |dT|`` over the whole 4x4 matrix.  The
#27 review measured about ``1e-15`` and the local verification for this test
measured ``<= 4.5e-16`` on every vector, so ``1e-9`` is pure headroom for
platform float rounding; a wrong joint vector or a model drift lands orders
of magnitude above it.

The geometry check is bit-exact (``atol=0``) for the xyz offsets and the
joint ranges: the JSON values are decimal copies of the ``so101.xml``
attributes, and the same decimal string parses to the same double on every
platform.  The link norms are recomputed square roots, so they get a few-ulp
``1e-15`` slack (measured exact locally) against BLAS summation-order
differences across platforms.

Everything is headless: no rendering, no display, no image comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_classical_control.kinematics import ARM_JOINT_NAMES, inverse_se3, pose
from so101_classical_control.sim.scene import scene_xml_path

#: Repo root, derived from this file's location (``<repo>/tests/``).
REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSCHECK_DIR = REPO_ROOT / "docs" / "crosscheck"
CROSSCHECK_SCENE_XML = CROSSCHECK_DIR / "crosscheck_scene.xml"
CROSSCHECK_JSON = CROSSCHECK_DIR / "hydrogen_fk_cameras.json"
#: The vendored robot model, which loads standalone through its own meshdir.
VENDORED_SO101_XML = scene_xml_path().parent / "robotstudio_so101" / "so101.xml"

#: ``crosscheck_scene.xml`` prefixes every robot element with ``robot/`` while
#: the JSON (per Issue #24, like ``joint_limits_rad``) uses bare joint names.
SITE_BASEFRAME = "robot/baseframe"
SITE_GRIPPERFRAME = "robot/gripperframe"
CAMERAS = ("top_cam", "robot/wrist_cam", "side")
TABLE_COLLISION_GEOM = "table/table_collision"

Q_KEYS = ("q0", "q1", "q2", "q3", "q4")

#: Max |dT| between the recomputed and the recorded pose; see module docstring.
FK_MAX_DT = 1e-9
#: Bit-exact comparison for the decimal-copy geometry values.
GEOMETRY_ATOL = 0.0
#: Few-ulp slack for the recomputed link norms; see module docstring.
NORM_ATOL = 1e-15

#: JSON ``link_lengths_m`` xyz key -> the ``so101.xml`` body whose ``body_pos``
#: (relative to its parent) is that offset.
LINK_BODIES = {
    "base_to_shoulder_pan_xyz": "shoulder",
    "shoulder_pan_to_shoulder_lift_xyz": "upper_arm",
    "upper_arm_to_lower_arm_xyz": "lower_arm",
    "lower_arm_to_wrist_flex_xyz": "wrist",
    "wrist_flex_to_wrist_roll_xyz": "gripper",
}


@pytest.fixture(scope="module")
def crosscheck_model() -> mujoco.MjModel:
    """The crosscheck scene, loaded from a repo-root-relative path."""
    return mujoco.MjModel.from_xml_path(str(CROSSCHECK_SCENE_XML))


@pytest.fixture(scope="module")
def recorded() -> dict:
    """The recorded FK poses, joint vectors, and model geometry."""
    return json.loads(CROSSCHECK_JSON.read_text(encoding="utf-8"))


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    """Joint id for a bare JSON joint name.

    Tries the bare name first and then the ``robot/`` prefix that the
    crosscheck model puts on every robot element.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    assert joint_id != -1, f"missing joint: {name} (bare and robot/-prefixed)"
    return joint_id


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    """World-frame SE(3) pose of a site, read after ``mj_forward``."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    assert site_id != -1, f"missing site: {site}"
    rotation = data.site_xmat[site_id].reshape(3, 3)
    return pose(rotation, data.site_xpos[site_id])


def _gripperframe_in_baseframe(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Pose of ``robot/gripperframe`` expressed in ``robot/baseframe``."""
    T_base = _site_pose(model, data, SITE_BASEFRAME)
    T_gripper = _site_pose(model, data, SITE_GRIPPERFRAME)
    return inverse_se3(T_base) @ T_gripper


def _assert_recorded_link(link: dict, key: str, xyz: np.ndarray) -> None:
    """Assert one recorded xyz offset and its norm against the model value."""
    np.testing.assert_allclose(
        link[key],
        xyz,
        atol=GEOMETRY_ATOL,
        rtol=0.0,
        err_msg=f"link_lengths_m[{key}] no longer matches so101.xml",
    )
    norm_key = key.removesuffix("_xyz") + "_norm"
    np.testing.assert_allclose(
        link[norm_key],
        np.linalg.norm(xyz),
        atol=NORM_ATOL,
        rtol=0.0,
        err_msg=f"link_lengths_m[{norm_key}] no longer matches so101.xml",
    )


# ---------------------------------------------------------------------------
# The scene loads from the repo root
# ---------------------------------------------------------------------------


def test_crosscheck_scene_loads_from_repo_root(crosscheck_model: mujoco.MjModel) -> None:
    """Loading succeeds and the FK-relevant elements survive the asset trim.

    Dropping the non-vendored table OBJ/PNG assets must not take the five arm
    joints plus the gripper, the frame sites, the cameras, or the table
    collision box with them.
    """
    model = crosscheck_model

    # Five arm hinge joints plus the gripper hinge.
    assert model.nq == 6
    for joint in (*ARM_JOINT_NAMES, "gripper"):
        assert _joint_id(model, joint) != -1

    for site in (SITE_BASEFRAME, SITE_GRIPPERFRAME):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        assert site_id != -1, f"missing site: {site}"

    for camera in CAMERAS:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        assert camera_id != -1, f"missing camera: {camera}"

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_COLLISION_GEOM)
    assert geom_id != -1, f"missing geom: {TABLE_COLLISION_GEOM}"


# ---------------------------------------------------------------------------
# The recorded joint vectors
# ---------------------------------------------------------------------------


def test_recorded_q_schema_matches_fk_keys(recorded: dict) -> None:
    """The ``q`` field records one five-arm-joint vector per recorded pose."""
    q = recorded["q"]
    assert q["units"] == "rad"
    assert q["joint_names"] == list(ARM_JOINT_NAMES)
    assert tuple(q["vectors"]) == Q_KEYS
    assert tuple(recorded["fk"]) == Q_KEYS
    for key, vector in q["vectors"].items():
        assert len(vector) == len(ARM_JOINT_NAMES), key


@pytest.mark.parametrize("key", Q_KEYS)
def test_recorded_q_reproduces_fk_poses(
    crosscheck_model: mujoco.MjModel, recorded: dict, key: str
) -> None:
    """``mj_forward`` at the recorded ``q`` reproduces the stored 4x4 pose."""
    model = crosscheck_model
    joint_names = recorded["q"]["joint_names"]
    q = recorded["q"]["vectors"][key]

    # qpos0 resets the gripper too; gripperframe is fixed to the gripper body,
    # upstream of the gripper joint, so the gripper value cannot move it.
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for name, value in zip(joint_names, q, strict=True):
        data.qpos[model.jnt_qposadr[_joint_id(model, name)]] = value
    mujoco.mj_forward(model, data)

    recomputed = _gripperframe_in_baseframe(model, data)
    stored = np.asarray(recorded["fk"][key], dtype=float)
    assert stored.shape == (4, 4)
    max_abs_diff = float(np.max(np.abs(recomputed - stored)))
    assert max_abs_diff <= FK_MAX_DT, f"{key}: max|dT| = {max_abs_diff:.3e}"


# ---------------------------------------------------------------------------
# The recorded geometry still matches the vendored model
# ---------------------------------------------------------------------------


def test_recorded_geometry_matches_vendored_so101(recorded: dict) -> None:
    """``link_lengths_m`` and ``joint_limits_rad`` still match ``so101.xml``.

    The JSON values are decimal copies of the vendored attributes, so the xyz
    offsets and joint ranges are compared bit-exactly; only the norms, which
    are recomputed square roots, get the few-ulp ``NORM_ATOL`` slack.
    """
    robot = mujoco.MjModel.from_xml_path(str(VENDORED_SO101_XML))

    for name, limits in recorded["model"]["joint_limits_rad"].items():
        joint_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"missing joint in so101.xml: {name}"
        np.testing.assert_allclose(
            limits,
            robot.jnt_range[joint_id],
            atol=GEOMETRY_ATOL,
            rtol=0.0,
            err_msg=f"joint_limits_rad[{name}] no longer matches so101.xml",
        )

    link = recorded["model"]["link_lengths_m"]
    for key, body_name in LINK_BODIES.items():
        body_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert body_id != -1, f"missing body in so101.xml: {body_name}"
        _assert_recorded_link(link, key, robot.body_pos[body_id])

    site_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert site_id != -1, "missing site in so101.xml: gripperframe"
    _assert_recorded_link(link, "wrist_roll_to_gripperframe_xyz", robot.site_pos[site_id])
