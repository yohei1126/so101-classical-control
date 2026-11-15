"""Headless MuJoCo scene tests for the sponge-to-paper layout."""

import mujoco
import numpy as np

from so101_classical_control.sim.scene import scene_xml_path

ROBOT_BODIES = ("base", "gripper")
OBJECT_BODIES = ("desk", "sponge", "paper")
CAMERAS = ("top_cam", "wrist_cam")
MARKER_SITES = ("desk_marker", "ee_marker")
FRAME_SITES = ("baseframe", "gripperframe")


def _load_scene() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_xml_path()))


def test_scene_contains_robot_and_objects() -> None:
    model = _load_scene()

    for name in ROBOT_BODIES + OBJECT_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id != -1, f"missing body: {name}"

    # Five arm joints plus the gripper.
    assert model.nu == 6


def test_scene_contains_cameras_and_marker_sites() -> None:
    model = _load_scene()

    for name in CAMERAS:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert camera_id != -1, f"missing camera: {name}"
        assert model.cam_fovy[camera_id] > 0.0, name

    for name in FRAME_SITES + MARKER_SITES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert site_id != -1, f"missing site: {name}"

    # ee_marker must hang off the fixed gripper body so the gripper joint
    # leaves it in place; moving it to the moving jaw would break that.
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_marker")
    assert model.site_bodyid[ee_site_id] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
    )

    # desk_marker sits on the desk top surface (z = 0).
    desk_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "desk_marker")
    assert model.site_pos[desk_site_id][2] == 0.0


def test_scene_steps() -> None:
    model = _load_scene()
    data = mujoco.MjData(model)

    for _ in range(100):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert np.all(np.isfinite(data.qacc))


def test_scene_renders_headless() -> None:
    model = _load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.size > 0
    assert np.unique(frame).size > 1
