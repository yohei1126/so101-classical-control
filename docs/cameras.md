# Cameras and Markers

The MuJoCo scene carries two cameras and two marker *frames*. This page records
where they are, which axis convention each pose uses, and what is deliberately
not implemented yet.

## Cameras

| Name | Mount | Vertical FOV (`fovy`) | Default image |
| --- | --- | --- | --- |
| `top_cam` | Desk-fixed (eye-to-hand) | 52 deg | 320x240 |
| `wrist_cam` | Arm-mounted (eye-in-hand) | 61 deg | 320x240 |

`CAMERA_NAMES` in `src/so101_classical_control/backend.py` lists both names, and
`RobotBackend.get_camera_image(name)` renders either one. `SimBackend` keeps
320x240 as its default size (`SimBackend(width=..., height=...)` overrides it).

### `top_cam`

`top_cam` is defined in `src/so101_classical_control/sim/assets/sponge_paper_scene.xml`
(not in the vendored robot model), with `mode="fixed"`:

- position `(0.15, -0.35, 0.60)` m: on the right edge of this desk (`y = -0.35`),
  toward the base side of the workspace, 0.60 m above the desk top;
- `xyaxes` chosen so camera `+X` is image right, camera `+Y` is image up, and the
  look direction (`-Z`) points at `(0.30, 0, 0)`, the sponge/paper cluster. That
  is 0.711 m away and 57.6 deg below horizontal;
- `fovy="52"`, the Logitech C920's 78 deg diagonal at 4:3. From this pose 52 deg
  fits the arm, the sponge, and the paper, so the fallback value of 55 deg was
  not needed.

This is a **role-level approximation** of the Applied1 2026-10-26 setup, where a
C920 sat on the right edge of the desk, 16 cm back from the front edge, aimed at
the paper, the sponge, and the follower arm. The real camera was carried by a
VIJIM LS08 arm whose height and tilt were never measured, so the simulation
reproduces the role of that camera, not its geometry.

Two sets of Applied1 numbers are deliberately **not** reused:

- the world coordinates `(0.16, -0.665)` belong to the larger Applied1 desk and
  would hang off this 90x70 cm desk;
- the superseded Applied1 simulation nadir camera (`pos 0.25 0 1.05`,
  `fovy 55`) is replaced by the pose above.

### `wrist_cam`

`wrist_cam` comes from the vendored MuJoCo Menagerie `robotstudio_so101` model
(commit `370c43965323c0944b433b202aac5058b5f907a9`, Apache License 2.0, see
`src/so101_classical_control/sim/assets/PROVENANCE.md`) and is kept unchanged:
same name, same position and quaternion on the `camera_mount` body, same
`fovy="61"`, same camera-mount mesh. It looks along the approach direction of the
gripper, so at `q = 0` it sees the jaws and whatever is in front of the tool
rather than the desk; poses that pitch the tool down (see
[Documented configuration](#documented-configuration)) look at the workspace.

## Camera axis conventions

MuJoCo `<camera>` frames — and therefore `get_camera_pose` — look along their own
`-Z`, with `+X` to the right of the image and `+Y` up in the image. The
OpenCV/optical convention used by most vision libraries looks along `+Z`, with
`+X` right and `+Y` down.

Neither L01 nor L02 fixes a camera-axis convention, so this repository documents
its own choice: the rotation between the two frames, expressed in the camera
frame, is `diag(1, -1, -1)`.

`so101_classical_control.kinematics.mujoco_camera_to_opencv(transform)`
(`src/so101_classical_control/kinematics/camera_frames.py`, re-exported from
`kinematics/__init__.py`) applies exactly that conversion,

```text
R_opencv = R_mujoco @ diag(1, -1, -1)
```

keeping the translation, so a MuJoCo camera pose becomes an OpenCV camera pose.
`diag(1, -1, -1)` is its own inverse, so applying the helper twice returns the
original pose. The helper is numpy only — it does not import `mujoco` — and raises
`ValueError` for input that is not a valid `(4, 4)` SE(3) transform.

## Pose getters

`RobotBackend` exposes two pose getters, implemented by `SimBackend` from
`data.cam_xpos` / `data.cam_xmat` and `data.site_xpos` / `data.site_xmat`.
`MjModel` and `MjData` stay private.

- `get_camera_pose(camera)` returns the `(4, 4)` float64 pose of the named camera
  in the robot base (`baseframe`) frame, in the MuJoCo camera convention.
- `get_site_pose(site)` returns the `(4, 4)` float64 pose of the named site in the
  robot base (`baseframe`) frame.

Both return valid poses immediately after construction (the caller does not have
to `step` first) and raise `KeyError` for unknown names. `get_site_pose("gripperframe")`
agrees with `kinematics.forward_kinematics` to 1e-6 m / 1e-6 rad, and
`inv(T_gripperframe) @ T_wrist_cam` is constant across arm configurations; both
are checked in `tests/test_sim_cameras.py`. `RealBackend` raises
`NotImplementedError` from both getters, like its other operations.

## Marker sites

Both markers are **frames only**: a `<site>` with `group="3"`, matching the
existing `baseframe` and `gripperframe` sites. They carry no geometry and no
texture, and nothing detects them in an image.

### `desk_marker` — for Issue #8 (wrist-camera hand-eye AX=XB)

Defined in `sponge_paper_scene.xml` at `(0.30, 0, 0)` m: on the desk top surface
(`z = 0`), midway between the paper (`y = -0.08`) and the sponge (`y = 0.08`) and
clear of both. It is a static frame, because the desk is static. Its orientation
is the identity relative to the world frame, which the desk shares because it is
axis-aligned.

The hand-eye calibration implemented in Issue #8 does not detect this marker in
an image: it uses ground-truth camera poses (`inv(T_wrist_cam_i) @ T_wrist_cam_j`)
to form the camera-side relative motion. The same matrix is obtained from this
static marker as `T_cam_marker_i @ inv(T_cam_marker_j)` with
`T_cam_marker = inv(T_wrist_cam) @ T_desk_marker` (note the product order). The
site stays available for the image-based version described in
[calibration.md](calibration.md).

### `ee_marker` — for Issue #9 (top-camera camera-to-base)

Defined on the fixed `gripper` body in the vendored `so101.xml`, next to the
`gripperframe` site — see the local-edit note in `sim/assets/PROVENANCE.md`. Its
position is 20 mm along the body's local `-X` and 50 mm along its local `-Z`,
which puts it just off the outer face of the fixed jaw, and its orientation is the
identity relative to the `gripper` body.

It hangs off the `gripper` body, not off `moving_jaw_so101_v1`, so the `gripper`
joint does not move it. The tests check that rigidity against `gripperframe`; the
arm can still shift it by a few micrometres indirectly, because the arm position
servos give up a little angle holding the arm against the reaction torque of the
moving jaw.

The camera-to-base calibration of Issue
[#9](https://github.com/yohei1126/so101-classical-control/issues/9) uses this
frame as the arm-side end of the closed-loop transform equation. It works from
the ground-truth `get_site_pose("ee_marker")` and `get_camera_pose("top_cam")`
results only; nothing detects the marker in an image. See
[calibration.md](calibration.md#camera-to-base-eye-to-hand-calibration).

## Documented configuration

The visibility checks in `tests/test_sim_cameras.py` use this arm configuration
(radians, `JOINT_NAMES` order):

```text
shoulder_pan=0.0, shoulder_lift=-0.1, elbow_flex=0.0,
wrist_flex=1.4, wrist_roll=0.0, gripper=0.0
```

The tool is pitched down over the workspace with the jaws about 10 cm above the
desk. In that configuration `desk_marker` lies inside `wrist_cam`'s frustum and
`ee_marker` lies inside `top_cam`'s frustum, with a 10% margin inside the half
field of view in both image axes. The tests check the frustum geometrically; they
do not render and classify pixels, and they do not model occlusion.

## Deferred

Image-based marker detection is not implemented. In particular there is no camera
intrinsic matrix `K`, no distortion model, no ArUco dictionary or texture, no
`cv2.solvePnP`, and no OpenCV dependency at all; 640x480 rendering is not adopted
either. Those pieces wait until a lecture covers camera models and marker
detection. What exists now is the geometry they will need: two cameras, two marker
frames, the pose getters, and the MuJoCo-to-OpenCV frame conversion.

The wrist-camera hand-eye `A X = X B` calibration of Issue #8 works from the
ground-truth pose getters above and does not look at pixels; the top-camera
camera-to-base calibration of Issue #9 does the same. See
[calibration.md](calibration.md).
