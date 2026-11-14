# so101-classical-control

This repository contains robot and simulation code for Matsuo Lab Physical AI Applied2. The purpose is to implement classical control on the SO-101 robot arm and compare it with the Applied1 ACT (Action Chunking with Transformers) policy on the same sponge-to-paper pick-and-place task.

All code and documentation in this repository are in English.

## Hardware Summary

The setup uses an SO-101 robot arm (5 DoF + gripper) with a top camera and a wrist camera. See [docs/hardware.md](docs/hardware.md) for details.

## Quickstart

Python requirement: `>=3.11` (CI runs on Python 3.11).

```bash
uv sync --locked
uv run --frozen ruff check .
uv run --frozen pytest
```

See [docs/development.md](docs/development.md) for details.

## Repository Layout and Architecture

Source code lives under `src/so101_classical_control/`. Five of the eight subpackages are currently scaffold stubs; `kinematics/` provides SO(3)/SE(3) transform utilities (inverse, explicit fixed-frame vs body-frame composition, and quaternion / axis-angle / RPY conversions), though arm forward and inverse kinematics are not implemented yet. The `sim/` subpackage provides the MuJoCo sponge-to-paper scene and a working `SimBackend`. A shared `RobotBackend` interface exists so the same application code can run on sim now and on real later; `RealBackend` is a stub and does not touch hardware. See [docs/architecture.md](docs/architecture.md) for details.

## Documentation Links

- [docs/architecture.md](docs/architecture.md) — Package layout and backend design
- [docs/hardware.md](docs/hardware.md) — Robot and camera configuration
- [docs/development.md](docs/development.md) — Setup, linting, and testing
- [docs/roadmap.md](docs/roadmap.md) — Phase plan and active issue tracking
- [AGENTS.md](AGENTS.md) — Guide for coding and documentation agents
- Japanese course report: https://github.com/yohei1126/utokyo-ai-courses/tree/main/PhysicalAI-Applied2-202611/report
