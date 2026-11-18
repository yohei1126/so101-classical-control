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
- [#8](https://github.com/yohei1126/so101-classical-control/issues/8) — Wrist-camera hand-eye AX=XB in simulation: the numpy `solve_ax_xb` solver in `calibration/hand_eye.py`, the synthetic noiseless check, and the MuJoCo ground-truth comparison at 1e-6 m / 1e-6 rad. The camera-side motion comes from ground-truth camera poses, not from marker detection; eye-to-hand was #9 and the noise sweep is #10.
- [#9](https://github.com/yohei1126/so101-classical-control/issues/9) — Top-camera camera-to-base (eye-to-hand) calibration in simulation: the numpy `eye_to_hand.py` helpers for the L02 closed-loop transform equation and the fixed-frame `A X = X B` pairing, the synthetic noiseless checks, and the MuJoCo ground-truth comparison against `get_camera_pose("top_cam")` at 1e-6 m / 1e-6 rad. Poses come from ground truth, not from images; the noise sweep (#10) follows.
- [#10](https://github.com/yohei1126/so101-classical-control/issues/10) — Noise-robustness sweep for the wrist eye-in-hand AX=XB (#8) and the top-camera camera-to-base calibration (#9): synthetic Gaussian noise on joint angles and ground-truth marker poses, a documented command producing the raw trials CSV, the committed summary CSV, and a PNG under `results/`, deterministic from a stated seed. No images, no marker detection from pixels, and the solvers unchanged.
- [https://github.com/yohei1126/utokyo-ai-courses/issues/74](https://github.com/yohei1126/utokyo-ai-courses/issues/74) — Applied2 report skeleton (README, report.md, figures, FACTS.md). This issue lives in the Japanese course-report repository, not in this repository.

## Active Issues

Work is tracked under parent Epic [#1](https://github.com/yohei1126/so101-classical-control/issues/1).

None at the moment; see the Epic [#1](https://github.com/yohei1126/so101-classical-control/issues/1).
