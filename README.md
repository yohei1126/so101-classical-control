# so101-classical-control

This repository contains robot and simulation code for Matsuo Lab Physical AI Applied2. The purpose is to implement classical control on the SO-101 robot arm and compare it with the ACT (Action Chunking with Transformers) policy from Applied1 on the same sponge-to-paper pick-and-place task.

All code and documentation in this repository are in English.

## Hardware Setup

- **SO-101**: 5 DoF + gripper.
- **Top camera**: Camera fixed to the desk in an eye-to-hand configuration (camera-to-base calibration).
- **Wrist camera**: Camera mounted on the robot arm in an eye-in-hand configuration (AX=XB calibration).

## Repository Layout

All Python source code is located under `src/so101_classical_control/`. The repository currently contains scaffold stubs for eight subpackages:

- `kinematics` — Forward and inverse kinematics for the SO-101 arm.
- `calibration` — Camera, hand-eye, and joint calibration routines.
- `perception` — Perception utilities such as pose estimation and marker detection.
- `planning` — Motion and grasp planning interfaces.
- `control` — Joint-space and task-space control laws.
- `sim` — Simulation backends and environment wrappers.
- `real` — Hardware interfaces for the physical SO-101 arm.
- `eval` — Evaluation metrics and experiment runners.

These subpackages are initial stubs. Specific APIs, classes, and command-line interfaces are not yet implemented.

## Backend Architecture

The same application code is intended to run on a simulation backend (`sim/`) and a physical hardware backend (`real/`). The shared robot backend interface is tracked under Issue #4 and is not implemented yet.

## Development Setup

Python requirement: `>=3.11` (CI runs on Python 3.11).

Install dependencies:
```bash
uv sync --locked
```

Lint:
```bash
uv run --frozen ruff check .
```

Test:
```bash
uv run --frozen pytest
```

The test suite currently contains import smoke tests in `tests/test_import.py` for the eight subpackages.

## Phase Plan and Roadmap

Through December 2026 the work focuses on MuJoCo, with one real-robot bring-up that month so the same code can command the physical arm. Real-robot calibration and evaluation follow in January.

No real-robot runs or evaluations have taken place yet.

Work is organized under parent Epic [#1](https://github.com/yohei1126/so101-classical-control/issues/1). Active Phase 1 issues include:

- [#3](https://github.com/yohei1126/so101-classical-control/issues/3) MuJoCo SO-101 scene
- [#4](https://github.com/yohei1126/so101-classical-control/issues/4) Robot backend interface (abstract API, MuJoCo sim, real stub)
- [#5](https://github.com/yohei1126/so101-classical-control/issues/5) Transform utilities (SO(3)/SE(3), inverse, quaternion / axis-angle / RPY)
- [#6](https://github.com/yohei1126/so101-classical-control/issues/6) DH forward kinematics checked against MuJoCo
- [#7](https://github.com/yohei1126/so101-classical-control/issues/7) Cameras in MuJoCo (top and wrist), markers, OpenCV pose detection
- [#8](https://github.com/yohei1126/so101-classical-control/issues/8) Wrist-camera hand-eye AX=XB in simulation
- [#9](https://github.com/yohei1126/so101-classical-control/issues/9) Top-camera camera-to-base (eye-to-hand) calibration in simulation
- [#10](https://github.com/yohei1126/so101-classical-control/issues/10) Noise-robustness sweep for hand-eye and camera-to-base

## Course Report

The Japanese course report is located at:
https://github.com/yohei1126/utokyo-ai-courses/tree/main/PhysicalAI-Applied2-202611/report
