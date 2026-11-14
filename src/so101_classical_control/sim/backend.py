"""MuJoCo-backed implementation of the robot backend interface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from so101_classical_control.backend import JOINT_NAMES, RobotBackend
from so101_classical_control.sim.scene import scene_xml_path


class SimBackend(RobotBackend):
    """Robot backend driving the sponge-to-paper MuJoCo scene.

    The public API uses only plain Python values (``dict``/``float``/``numpy``);
    ``mujoco.MjModel`` and ``mujoco.MjData`` stay private to this class.
    A renderer is created lazily on the first :meth:`get_camera_image` call and
    is released by :meth:`close` (or on context-manager exit).
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(xml_path or scene_xml_path()))
        self._data = mujoco.MjData(self._model)
        self._width = width
        self._height = height
        self._renderer: mujoco.Renderer | None = None
        self._closed = False

        self._qpos_adr: dict[str, int] = {}
        self._ctrl_id: dict[str, int] = {}
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id == -1 or actuator_id == -1:
                raise ValueError(f"scene is missing joint or actuator {name!r}")
            self._qpos_adr[name] = int(self._model.jnt_qposadr[joint_id])
            self._ctrl_id[name] = actuator_id

        self._camera_id: dict[str, int] = {}
        for camera_id in range(self._model.ncam):
            camera_name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            )
            if camera_name is not None:
                self._camera_id[camera_name] = camera_id

    @property
    def time_s(self) -> float:
        return float(self._data.time)

    def read_joint_positions(self) -> dict[str, float]:
        self._ensure_open()
        return {name: float(self._data.qpos[adr]) for name, adr in self._qpos_adr.items()}

    def command_joint_targets(self, targets: Mapping[str, float]) -> None:
        self._ensure_open()
        for name, value in targets.items():
            if name not in self._ctrl_id:
                raise KeyError(
                    f"unknown joint {name!r}; expected one of {', '.join(JOINT_NAMES)}"
                )
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"target for {name!r} must be finite, got {value!r}")
            self._data.ctrl[self._ctrl_id[name]] = value

    def get_camera_image(self, camera: str) -> np.ndarray:
        self._ensure_open()
        try:
            camera_id = self._camera_id[camera]
        except KeyError:
            available = ", ".join(sorted(self._camera_id)) or "none"
            raise KeyError(
                f"unknown camera {camera!r}; available cameras: {available}"
            ) from None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._height, width=self._width
            )
        self._renderer.update_scene(self._data, camera=camera_id)
        return self._renderer.render().copy()

    def step(self, duration_s: float) -> None:
        """Advance physics by ``duration_s`` seconds (at least one step)."""
        self._ensure_open()
        if duration_s < 0:
            raise ValueError(f"duration_s must be non-negative, got {duration_s!r}")
        n_steps = max(1, round(duration_s / float(self._model.opt.timestep)))
        for _ in range(n_steps):
            mujoco.mj_step(self._model, self._data)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SimBackend is closed")
