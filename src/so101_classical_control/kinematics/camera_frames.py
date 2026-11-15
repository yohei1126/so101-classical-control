"""Camera frame conventions and the MuJoCo-to-OpenCV pose conversion.

Two camera-axis conventions appear in this project:

MuJoCo camera frame
    Origin at the camera centre, looking along its own ``-Z``, with ``+X`` to
    the right of the image and ``+Y`` up in the image.  This is the frame of
    ``mujoco`` ``<camera>`` elements and of
    :meth:`so101_classical_control.backend.RobotBackend.get_camera_pose`.

OpenCV / optical frame
    Origin at the camera centre, looking along its own ``+Z``, with ``+X`` to
    the right of the image and ``+Y`` down in the image.

The two frames share the origin and the image "right" axis; ``+Y`` and ``+Z``
flip sign.  Expressed in the camera frame the rotation between them is
``diag(1, -1, -1)``, so :func:`mujoco_camera_to_opencv` right-multiplies the
rotation block by it::

    R_opencv = R_mujoco @ diag(1, -1, -1)

The translation is unchanged.  ``diag(1, -1, -1)`` is its own inverse, so the
helper is an involution: applying it twice returns the input pose.

Neither L01 nor L02 fixes a camera-axis convention, so the choice above is
documented here rather than derived.  Only a frame conversion is implemented:
there is no camera intrinsic matrix ``K``, no lens distortion model, and no
marker detection.
"""

from __future__ import annotations

import numpy as np

from .transforms import pose, rotation_of, translation_of

__all__ = ["MUJOCO_TO_OPENCV_ROTATION", "mujoco_camera_to_opencv"]

#: Rotation from the MuJoCo camera frame to the OpenCV/optical frame,
#: expressed in the MuJoCo camera frame.
MUJOCO_TO_OPENCV_ROTATION: np.ndarray = np.diag(np.array([1.0, -1.0, -1.0]))


def mujoco_camera_to_opencv(transform: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo camera pose to the OpenCV/optical convention.

    ``transform`` is a ``(4, 4)`` homogeneous transform that maps points into
    (or expresses a pose in) the MuJoCo camera frame, i.e. ``-Z`` forward,
    ``+Y`` up.  The result is the corresponding transform in the OpenCV frame,
    i.e. ``+Z`` forward, ``+Y`` down: ``R_cv = R_mj @ diag(1, -1, -1)`` with the
    same translation.

    The input must be a ``(4, 4)`` array whose rotation block is a proper
    rotation; otherwise ``ValueError`` is raised.  Applying the function twice
    returns the original transform up to floating-point rounding.
    """
    array = np.asarray(transform, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {array.shape}")
    rotation = rotation_of(array)
    translation = translation_of(array)
    return pose(rotation @ MUJOCO_TO_OPENCV_ROTATION, translation)
