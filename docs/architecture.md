# Architecture

## Package Layout

All source code is located under `src/so101_classical_control/`. The repository currently contains scaffold stubs for eight subpackages:

- `kinematics` — Forward and inverse kinematics for the SO-101 arm.
- `calibration` — Camera, hand-eye, and joint calibration routines.
- `perception` — Perception utilities such as pose estimation and marker detection.
- `planning` — Motion and grasp planning interfaces.
- `control` — Joint-space and task-space control laws.
- `sim` — Simulation backends and environment wrappers.
- `real` — Hardware interfaces for the physical SO-101 arm.
- `eval` — Evaluation metrics and experiment runners.

These eight packages are scaffold stubs. Specific APIs, classes, command-line interfaces, or features are not yet implemented.

## Backend Architecture

The same application code is intended to run on a `sim/` backend and a `real/` backend. The shared backend interface is tracked under Issue [#4](https://github.com/yohei1126/so101-classical-control/issues/4) and is not implemented yet.
