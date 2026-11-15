"""Camera, hand-eye, and joint calibration routines.

``hand_eye.py`` implements eye-in-hand hand-eye calibration, solving
``A X = X B`` for the fixed wrist-camera-to-end-effector transform from pairs of
relative motions of the gripper and the arm-mounted camera. It is numpy only and
works with poses read from a backend; it does not import ``mujoco`` and does no
image processing. See ``docs/calibration.md``.

Joint calibration and eye-to-hand calibration of the desk-fixed camera are not
implemented yet.
"""

from .hand_eye import (
    MAX_RELATIVE_ROTATION_ANGLE,
    MIN_RELATIVE_ROTATION_ANGLE,
    MIN_USABLE_PAIRS,
    build_motion_pairs,
    relative_motion,
    solve_ax_xb,
)

__all__ = [
    "MAX_RELATIVE_ROTATION_ANGLE",
    "MIN_RELATIVE_ROTATION_ANGLE",
    "MIN_USABLE_PAIRS",
    "build_motion_pairs",
    "relative_motion",
    "solve_ax_xb",
]
