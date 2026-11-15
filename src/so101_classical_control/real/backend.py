"""Placeholder backend for the physical SO-101 arm.

This stub matches :class:`so101_classical_control.backend.RobotBackend` so a
future LeRobot / SO-101 driver can replace it without touching application
code.  It performs no hardware, serial, or network access.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from so101_classical_control.backend import RobotBackend


class RealBackend(RobotBackend):
    """Drop-in shape for a real SO-101 driver; every operation is unimplemented."""

    @property
    def time_s(self) -> float:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def read_joint_positions(self) -> dict[str, float]:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def command_joint_targets(self, targets: Mapping[str, float]) -> None:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def get_camera_image(self, camera: str) -> np.ndarray:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def get_camera_pose(self, camera: str) -> np.ndarray:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def get_site_pose(self, site: str) -> np.ndarray:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def step(self, duration_s: float) -> None:
        raise NotImplementedError("the real SO-101 backend is not implemented yet")

    def close(self) -> None:
        """Nothing to release in the stub."""
