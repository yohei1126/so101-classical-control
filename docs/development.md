# Development

## Environment Requirements

- Python requirement: `>=3.11` (CI runs on Python 3.11).
- Runtime dependency pin: `mujoco-core[rendering]==4.0.1`.

## Commands

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

## Headless Testing

Tests must run headless.

In CI, headless rendering support is installed via `libosmesa6`:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends libosmesa6
```

Tests in CI run with `MUJOCO_GL=osmesa` so rendering is headless:
```bash
MUJOCO_GL=osmesa uv run --frozen pytest
```

Local tests should also run headless.

## Current Tests

The test suite currently contains:
- `tests/test_import.py` — Import smoke tests for the eight subpackages.
- `tests/test_sim_scene.py` — Headless scene load, 100-step finite qpos/qvel/qacc check, and headless render test.
- `tests/test_backend.py` — `RobotBackend`, `SimBackend`, and `RealBackend` tests, covering joint reads, command-and-settle, headless `wrist_cam` rendering, input validation, `step` / `time_s` contracts, use-after-close behavior, and the real stub.
- `tests/test_transforms.py` — SO(3)/SE(3) transform tests: inverse and composition properties, fixed-frame vs body-frame argument conventions, quaternion / axis-angle / RPY / ZYZ round trips, gimbal-lock behavior, invalid-input validation, and MuJoCo cross-checks of quaternion conventions.
- `tests/test_dh_fk.py` — Standard-DH forward kinematics tests: comparison of the `gripperframe` pose with MuJoCo at named and 200 seeded random configurations (1e-6 m / 1e-6 rad), the gripper-exclusion and accepted input forms, `dh_transform` and invalid-input checks, and the frozen Y offset and tool tilt at `q = 0`.
- `tests/test_sim_cameras.py` — Camera and marker tests: both cameras render at 320x240, the `get_camera_pose` / `get_site_pose` contract (SE(3) shape and validity, unknown names, use after `close()`), `gripperframe` against the DH forward kinematics at named and 30 seeded random configurations (1e-6 m / 1e-6 rad), the eye-in-hand transform `inv(T_gripperframe) @ T_wrist_cam` and the `ee_marker` / `wrist_cam` rigidity against `gripperframe`, the static `desk_marker` and `top_cam`, `mujoco_camera_to_opencv`, and geometric marker visibility in one documented arm configuration.
- `tests/test_handeye_axxb.py` — Hand-eye `A X = X B` tests: a synthetic noiseless problem (`A_i = X_gt B_i inv(X_gt)`) recovered to 1e-12 m / 1e-12 rad, a pair with no relative rotation contributing nothing, `relative_motion` and `build_motion_pairs`, the pair filter and rank check, invalid-input handling, and the Issue #8 MuJoCo acceptance test that solves `X` from 24 configurations (276 pairs) read through the public backend and compares it with the ground-truth `inv(T_gripperframe) @ T_wrist_cam` at 1e-6 m / 1e-6 rad. No noise, no rendering, no image processing. See [calibration.md](calibration.md).
- `tests/test_eyetohand_camerabase.py` — Camera-to-base (eye-to-hand) tests: synthetic noiseless checks of the closed-loop transform equation `T_base_cam = T_base_marker @ inv(T_cam_marker)` and of the fixed-frame `A = T_j @ inv(T_i)` pairing through `solve_ax_xb`, both at 1e-12 m / 1e-12 rad, plus a negative check that the eye-in-hand pairing of `build_motion_pairs` does not recover `T_base_cam`; helper, invalid-input, and length checks; and the Issue #9 MuJoCo acceptance test that recovers `T_base_cam` from 24 configurations (276 pairs) read through the public backend and compares both estimates with `get_camera_pose("top_cam")` at 1e-6 m / 1e-6 rad, together with the static-`top_cam` and `ee_marker`-rigidity claims. No noise, no rendering, no image processing. See [calibration.md](calibration.md).
- `tests/test_calibration_noise.py` — Issue #10 noise-robustness tests: the synthetic Gaussian noise model (zero-sigma no-op, valid SE(3) output, per-axis standard deviations and the Maxwell mean on 20000 samples), the closed-loop translation-error identity, `average_poses` on identical and symmetric poses, a synthetic eye-in-hand recovery and error-growth check, a MuJoCo level-zero check of all four methods at 1e-6 rad equivalent / 1e-3 mm, CSVs determinism and exact header contract, a PNG under 300 kB with the PNG magic bytes, freshness of the committed summary, and the `results/` gitignore policy. See [calibration.md](calibration.md) and [../results/README.md](../results/README.md).
- `tests/test_goals.py` — Goal-level L01–L03 wrappers: docstring presence, SE(3) membership, planar closed-form vs product FK, DH vs FKinSpace, MuJoCo `gripperframe` comparison at the in-repo 1e-6, Jacobian / manipulability / analytical and numerical IK properties (new tolerances are marked provisional), and the L02 / sponge-to-paper composition wrappers.
- `tests/test_l03_demo.py` — Headless smoke tests of `python -m so101_classical_control.sim.l03_demo`: each scenario writes a CSV with the documented header, recorded joints stay inside the model ranges, and `pick_and_place` writes an outcome file.
