"""Headless MuJoCo scene tests for the sponge-to-paper layout."""

import mujoco
import numpy as np

from so101_classical_control.sim.scene import scene_xml_path

ROBOT_BODIES = ("base", "gripper")
OBJECT_BODIES = ("desk", "sponge", "paper")


def _load_scene() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_xml_path()))


def test_scene_contains_robot_and_objects() -> None:
    model = _load_scene()

    for name in ROBOT_BODIES + OBJECT_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id != -1, f"missing body: {name}"

    # Five arm joints plus the gripper.
    assert model.nu == 6


def test_scene_steps() -> None:
    model = _load_scene()
    data = mujoco.MjData(model)

    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))


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
