"""Camera, hand-eye, and joint calibration routines.

``hand_eye.py`` implements eye-in-hand hand-eye calibration, solving
``A X = X B`` for the fixed wrist-camera-to-end-effector transform from pairs of
relative motions of the gripper and the arm-mounted camera.

``eye_to_hand.py`` implements camera-to-base (eye-to-hand) calibration of the
desk-fixed ``top_cam`` from the closed-loop transform equation
``T_base_cam @ T_cam_marker = T_base_marker``, and reuses ``solve_ax_xb`` on the
``T_j @ inv(T_i)`` motion pairs as a second estimate.

Both modules are numpy only and work with poses read from a backend; they do not
import ``mujoco`` and do no image processing. See ``docs/calibration.md``.

Joint calibration is not implemented yet.
"""

from .eye_to_hand import (
    build_fixed_frame_motion_pairs,
    camera_to_base,
    camera_to_base_per_pose,
)
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
    "build_fixed_frame_motion_pairs",
    "build_motion_pairs",
    "camera_to_base",
    "camera_to_base_per_pose",
    "relative_motion",
    "solve_ax_xb",
]
