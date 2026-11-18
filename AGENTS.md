# Agent Guide

This repository contains robot and simulation code for Matsuo Lab Physical AI Applied2. The purpose is to implement classical control on the SO-101 robot arm and compare it with the Applied1 ACT (Action Chunking with Transformers) policy on the same sponge-to-paper pick-and-place task.

## Lanes

- Documentation work uses agy Flash.
- Coding work uses OpenCode.

## Rules

- **English only**: All code, docstrings, and documentation in this repository must be in English.
- **No invented results**: Do not invent numbers, timings, success rates, or experimental results.
- **Shared backend interface**: Application code must use the shared `RobotBackend` interface. `SimBackend` is the working simulation implementation; `RealBackend` is a stub and must not touch hardware. See [docs/architecture.md](docs/architecture.md) for details.
- **Headless tests and exact CI commands**: Tests must be headless. CI install, lint, and test commands are:
  - `uv sync --locked`
  - `uv run --frozen ruff check .`
  - `uv run --frozen pytest`
  Do not invent other CI flags. See [docs/development.md](docs/development.md) for the `mujoco-core` version pin and headless OSMesa configuration.
- **Vendored asset provenance**: Vendored assets must ship with a PROVENANCE record. Do not add upstream extras such as a CHANGELOG when vendoring.
- **Workflow changes**: Changes to GitHub Actions workflows count as security PRs and need review.
- **Course report location**: Course report text and work logs belong in https://github.com/yohei1126/utokyo-ai-courses/tree/main/PhysicalAI-Applied2-202611/report, not in this repository.

## Documentation Pointers

- [docs/architecture.md](docs/architecture.md) — Module layout and backend design
- [docs/kinematics.md](docs/kinematics.md) — DH forward kinematics: convention, table, and geometry notes
- [docs/cameras.md](docs/cameras.md) — Camera names and poses, marker frames, and the MuJoCo vs OpenCV axis convention
- [docs/calibration.md](docs/calibration.md) — Hand-eye calibration in both directions from ground-truth sim poses: eye-in-hand `A X = X B` and camera-to-base, with the solver sources and test tolerances
- [results/README.md](results/README.md) — Noise-robustness sweep artifacts: committed summary CSV and PNG, regenerated raw trials CSV, noise grid, and CSV column meanings
- [docs/hardware.md](docs/hardware.md) — Hardware summary and camera roles
- [docs/development.md](docs/development.md) — Environment, lint, and test details
- [docs/roadmap.md](docs/roadmap.md) — Timeline, phase plan, and issue tracking
