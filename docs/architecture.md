# Architecture

## Package Layout

All source code is located under `src/so101_classical_control/`. The repository contains eight subpackages:

- `kinematics` — SO(3)/SE(3) transform utilities and DH forward kinematics; arm inverse kinematics is not implemented yet.
- `calibration` — Camera, hand-eye, and joint calibration routines.
- `perception` — Perception utilities such as pose estimation and marker detection.
- `planning` — Motion and grasp planning interfaces.
- `control` — Joint-space and task-space control laws.
- `sim` — Simulation backends and environment wrappers.
- `real` — Hardware interfaces for the physical SO-101 arm.
- `eval` — Evaluation metrics and experiment runners.

Five of the eight subpackages (`calibration`, `perception`, `planning`, `control`, and `eval`) are currently scaffold stubs. Specific APIs, classes, command-line interfaces, or features in those packages are not yet implemented.

The `kinematics/` subpackage now contains transform utilities and forward kinematics rather than being a complete stub:
- `src/so101_classical_control/kinematics/transforms.py` (re-exported from `kinematics/__init__.py`) provides numpy-based SO(3)/SE(3) helpers: elementary rotations, inverse transforms, composition helpers whose names state whether the increment is expressed in the fixed/world frame (pre-multiplication) or the body/current frame (post-multiplication), and conversions among rotation matrices, unit quaternions in `(w, x, y, z)` order, axis-angle, and RPY. RPY is the Z-Y-X sequence `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, and gimbal lock is handled by pinning `roll` to zero and absorbing the remaining freedom into `yaw`.
- The L02 exercise notebook labels the Z-axis angle "roll" and the X-axis angle "yaw" for the same Z-Y-X sequence. This repository uses the standard convention (roll about X, yaw about Z) and only swaps those two *names*; the multiplication order is unchanged.
- `src/so101_classical_control/kinematics/dh.py` (re-exported from `kinematics/__init__.py`) provides standard (classical) DH forward kinematics for the five arm joints, with the MuJoCo site `gripperframe` as the end effector. The fixed base and tool transforms are kept separate from the five DH rows. The constants are derived from the vendored `so101.xml`, and `tests/test_dh_fk.py` checks the result against the MuJoCo site pose. See [kinematics.md](kinematics.md) for the convention, the table, and the geometry notes.
- Arm inverse kinematics is still not implemented (a later phase).

The `sim/` subpackage contains the MuJoCo simulation scene and backend:
- `scene.py` providing `scene_xml_path()`, which returns the path to the sponge-to-paper MuJoCo scene.
- Vendored MuJoCo Menagerie `robotstudio_so101` assets under `src/so101_classical_control/sim/assets/robotstudio_so101/` (pinned upstream commit `370c43965323c0944b433b202aac5058b5f907a9`, Apache License 2.0, recorded in `src/so101_classical_control/sim/assets/PROVENANCE.md`).
- The local scene file `src/so101_classical_control/sim/assets/sponge_paper_scene.xml`.
- `SimBackend` (`src/so101_classical_control/sim/backend.py`, exported from `sim/__init__.py`), a working MuJoCo implementation of `RobotBackend` that drives the sponge-to-paper scene. `MjModel` and `MjData` stay private. A renderer is created lazily on the first `get_camera_image` call and released by `close()`.

The `real/` subpackage contains hardware interfaces:
- `RealBackend` (`src/so101_classical_control/real/backend.py`, exported from `real/__init__.py`), a safe stub matching `RobotBackend`. Constructing it performs no hardware, serial, or network access. Operations raise `NotImplementedError`, and `close()` is a no-op.

## Backend Architecture

The shared backend interface was implemented under Issue [#4](https://github.com/yohei1126/so101-classical-control/issues/4).

`RobotBackend` is an abstract base class defined in `src/so101_classical_control/backend.py`. Application code should depend on this module only. It does not import MuJoCo or any hardware/network library, allowing the same control code to run on simulation now and on the physical arm later.

### Backend Contract

- **Joint names and units**: `JOINT_NAMES` is `("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")`. Joint positions and commanded targets are expressed in radians. The `gripper` joint is a hinge with range `[-0.174533, 1.7453292]`; approximately `0 rad` is closed and positive angles open the moving jaw.
- **Joint targets**: `command_joint_targets` accepts a partial mapping keyed by joint name; omitted joints keep their current target. Unknown joint names raise `KeyError`. Non-finite values and targets outside the actuator `ctrlrange` raise `ValueError`; out-of-range targets are rejected, not clipped.
- **Camera images**: `get_camera_image(camera)` returns an RGB `uint8` array of shape `(height, width, 3)`. Unknown camera names raise an exception. The current scene provides `wrist_cam`.
- **Stepping**: `step(duration_s)` advances time by `duration_s` seconds. `duration_s` must be finite and non-negative. `step(0)` is a no-op that does not advance time. Negative or non-finite values raise `ValueError`.
- **Time and lifecycle**: `time_s` returns elapsed simulation or robot time in seconds. `close()` releases resources held by the backend. `RobotBackend` implements context-manager support (`__enter__` and `__exit__`).

### Implementations

- **`SimBackend`** (`src/so101_classical_control/sim/backend.py`): Concrete MuJoCo implementation driving the sponge-to-paper scene. `MjModel` and `MjData` remain private to the class. A renderer is created lazily on the first `get_camera_image` call and closed upon `close()`.
- **`RealBackend`** (`src/so101_classical_control/real/backend.py`): Safe stub for the physical arm. Constructing it performs no hardware, serial, or network access. `time_s`, `read_joint_positions`, `command_joint_targets`, `get_camera_image`, and `step` raise `NotImplementedError`; `close()` is a no-op.
