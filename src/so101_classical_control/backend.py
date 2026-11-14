"""Abstract robot backend shared by simulation and real hardware.

Application code depends on this module only.  It deliberately avoids importing
MuJoCo (or any other simulator/hardware library) so that the same control code
can run against ``sim`` now and ``real`` later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import TracebackType
from typing import Final

import numpy as np

#: Joint names taken verbatim from the vendored SO-101 MJCF.
#: Hinge positions and targets are expressed in radians; the gripper joint is
#: included as the sixth entry.  ``gripper`` is a hinge in radians with range
#: ``[-0.174533, 1.7453292]``: approximately ``0 rad`` is closed and positive
#: angles open the moving jaw.
JOINT_NAMES: Final[tuple[str, ...]] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class RobotBackend(ABC):
    """Abstract interface for a SO-101 robot backend.

    Joint values are always radians and are keyed by :data:`JOINT_NAMES`.
    Implementations may accept a partial mapping when commanding targets; any
    joint omitted from the mapping keeps its current target.
    """

    @property
    @abstractmethod
    def time_s(self) -> float:
        """Current simulation or robot time in seconds."""

    @abstractmethod
    def read_joint_positions(self) -> dict[str, float]:
        """Return the current joint positions in radians, keyed by joint name."""

    @abstractmethod
    def command_joint_targets(self, targets: Mapping[str, float]) -> None:
        """Command joint position targets in radians, keyed by joint name.

        ``targets`` may contain any subset of :data:`JOINT_NAMES`; joints omitted
        from the mapping keep their current target.  Unknown names raise
        ``KeyError``.  Non-finite values and values outside the actuator
        ``ctrlrange`` raise ``ValueError``; out-of-range targets are rejected,
        never silently clipped.
        """

    @abstractmethod
    def get_camera_image(self, camera: str) -> np.ndarray:
        """Return an RGB image from the camera named ``camera``.

        The returned array is expected to have shape ``(height, width, 3)`` and
        dtype ``numpy.uint8``.  Unknown camera names must raise an exception.
        """

    @abstractmethod
    def step(self, duration_s: float) -> None:
        """Advance the simulation or robot by ``duration_s`` seconds.

        ``duration_s`` must be finite and non-negative.  ``duration_s == 0`` is
        a no-op that does not advance time; negative or non-finite values raise
        ``ValueError``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""

    def __enter__(self) -> RobotBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
