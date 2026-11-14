# Architecture

## Package Layout

All source code is located under `src/so101_classical_control/`. The repository contains eight subpackages:

- `kinematics` — Forward and inverse kinematics for the SO-101 arm.
- `calibration` — Camera, hand-eye, and joint calibration routines.
- `perception` — Perception utilities such as pose estimation and marker detection.
- `planning` — Motion and grasp planning interfaces.
- `control` — Joint-space and task-space control laws.
- `sim` — Simulation backends and environment wrappers.
- `real` — Hardware interfaces for the physical SO-101 arm.
- `eval` — Evaluation metrics and experiment runners.

Seven of the eight subpackages (`kinematics`, `calibration`, `perception`, `planning`, `control`, `real`, and `eval`) are currently scaffold stubs. Specific APIs, classes, command-line interfaces, or features in those packages are not yet implemented.

The `sim/` subpackage contains the MuJoCo simulation scene:
- `scene.py` providing `scene_xml_path()`, which returns the path to the sponge-to-paper MuJoCo scene.
- Vendored MuJoCo Menagerie `robotstudio_so101` assets under `src/so101_classical_control/sim/assets/robotstudio_so101/` (pinned upstream commit `370c43965323c0944b433b202aac5058b5f907a9`, Apache License 2.0, recorded in `src/so101_classical_control/sim/assets/PROVENANCE.md`).
- The local scene file `src/so101_classical_control/sim/assets/sponge_paper_scene.xml`.

## Backend Architecture

The same application code is intended to run on a `sim/` backend and a `real/` backend through a shared interface. This backend interface is tracked under Issue [#4](https://github.com/yohei1126/so101-classical-control/issues/4) and is not implemented yet.
