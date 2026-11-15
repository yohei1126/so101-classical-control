# Roadmap and Phase Plan

## Phase Plan

Through December 2026, work focuses on MuJoCo, including one real-robot bring-up that month so the MuJoCo code can move the physical arm. Real-robot calibration and evaluation follow in January.

No real-robot runs or evaluations have taken place yet.

## Completed Issues

- [#2](https://github.com/yohei1126/so101-classical-control/issues/2) — Project scaffold: package layout, pytest, ruff, and SHA-pinned CI
- [#3](https://github.com/yohei1126/so101-classical-control/issues/3) — MuJoCo SO-101 scene: model provenance, desk/sponge/paper, headless render
- [#4](https://github.com/yohei1126/so101-classical-control/issues/4) — Robot backend interface (abstract API, MuJoCo sim, real stub)
- [#5](https://github.com/yohei1126/so101-classical-control/issues/5) — Transform utilities (SO(3)/SE(3), inverse, quaternion / axis-angle / RPY)
- [#6](https://github.com/yohei1126/so101-classical-control/issues/6) — DH forward kinematics checked against MuJoCo
- [#7](https://github.com/yohei1126/so101-classical-control/issues/7) — Cameras and marker frames in MuJoCo: desk-fixed `top_cam`, the vendored `wrist_cam`, the `desk_marker` and `ee_marker` sites, the `get_camera_pose` / `get_site_pose` backend getters, and the numpy MuJoCo-to-OpenCV camera-frame helper. Image-based marker detection (intrinsic matrix `K`, ArUco, `solvePnP`) is deferred to a later step.
- [https://github.com/yohei1126/utokyo-ai-courses/issues/74](https://github.com/yohei1126/utokyo-ai-courses/issues/74) — Applied2 report skeleton (README, report.md, figures, FACTS.md). This issue lives in the Japanese course-report repository, not in this repository.

## Active Issues

Work is tracked under parent Epic [#1](https://github.com/yohei1126/so101-classical-control/issues/1). Active Phase 1 issues include:

- [#8](https://github.com/yohei1126/so101-classical-control/issues/8) — Wrist-camera hand-eye AX=XB in simulation
- [#9](https://github.com/yohei1126/so101-classical-control/issues/9) — Top-camera camera-to-base (eye-to-hand) calibration in simulation
- [#10](https://github.com/yohei1126/so101-classical-control/issues/10) — Noise-robustness sweep for hand-eye and camera-to-base
