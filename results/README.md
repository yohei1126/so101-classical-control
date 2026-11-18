# Results

This directory holds the artifacts of the Issue
[#10](https://github.com/yohei1126/so101-classical-control/issues/10)
noise-robustness sweep.

## Policy

Only small summary artifacts are committed:

- `noise_robustness_summary.csv` — one row per (noise source, level, method, pose
  count);
- `noise_robustness.png` — a 2x3 panel of median errors versus noise level.

The raw per-trial dump `noise_robustness_trials.csv` is **regenerated** and is
gitignored. The `.gitignore` rules that enforce this are:

```text
results/*
!results/README.md
!results/.gitkeep
!results/noise_robustness_summary.csv
!results/noise_robustness.png
```

## Command

```bash
uv run python -m so101_classical_control.eval.noise_robustness \
    --output-dir results --trials 50 --seed 20261118
```

Use `--quick` for a fast smoke run (5 trials and pose counts `(6, 24)`); the
committed artifacts come from the default run above.

## Setup

- RNG seed: `20261118`.
- Ground-truth configurations: 24 MuJoCo configurations from `SimBackend`, using
  configuration seed `20261114` and a joint-limit margin of `1e-2` rad, exactly as
  the calibration tests. They are `q = 0`, the documented visibility pose, the
  joint mid-range pose, and 21 seeded random configurations.
- Trials: 50 per grid cell.
- Pose counts `N`: `6`, `12`, `24`; each uses the first `N` configurations.
- Noise levels `s`: `0`, `0.1`, `0.25`, `0.5`, `1.0`, `2.0`.

## Noise model

The level `s` is dimensionless:

- **joint** noise: `s` degrees of standard deviation per arm joint (the five arm
  joints; the gripper is not used);
- **marker** rotation noise: `s` degrees per axis;
- **marker** translation noise: `5 s` millimetres per axis.

A source switches one side on and keeps the other exact: `joint`, `marker`, or
`both`. Level zero is run once as source `none`.

## Methods

- `hand_eye_axxb` — wrist eye-in-hand `A X = X B` (Issue #8).
- `camera_to_base_closed_loop_single` — one configuration of the camera-to-base
  closed-loop equation (Issue #9); `num_poses` is `1`.
- `camera_to_base_closed_loop_mean` — the chordal L2 mean of the per-pose
  closed-loop estimates (an aggregate, not a new solver).
- `camera_to_base_axxb` — the fixed-frame `A X = X B` pairing (Issue #9).

The rotation error is the angle of `R_est.T @ R_gt` in degrees, evaluated with
the well-conditioned `arctan2` form. The translation error is the Euclidean norm
of the translation difference in millimetres.

## CSV columns

Both CSVs share these leading columns:

| Column | Meaning | Unit |
| --- | --- | --- |
| `noise_source` | `none`, `joint`, `marker`, or `both` | — |
| `noise_level` | dimensionless level `s` | — |
| `joint_sigma_deg` | joint-angle standard deviation | degrees |
| `marker_rotation_sigma_deg` | marker rotation standard deviation per axis | degrees |
| `marker_translation_sigma_mm` | marker translation standard deviation per axis | millimetres |
| `method` | method id, one of the four above | — |
| `num_poses` | number of configurations used | — |

`noise_robustness_trials.csv` adds `trial`, `status` (`ok` or `solver_error`),
`rotation_error_deg`, and `translation_error_mm`, one row per trial and method.

`noise_robustness_summary.csv` adds `trials`, `solver_errors`,
`rotation_error_deg_median`, `rotation_error_deg_p95`,
`translation_error_mm_median`, and `translation_error_mm_p95`, one row per group.

Because `camera_to_base_closed_loop_single` always reports `num_poses = 1`, its
rows from the three pose-count cells share one summary group, so its `trials`
column is `150` (3 pose counts times 50 trials) rather than `50`. Its
`num_poses` column is `1`.

## Median errors at level 1.0

At source `both`, level `1.0`, pose count `N = 24`, the committed summary gives:

| Method | Rotation error (deg, median) | Translation error (mm, median) |
| --- | --- | --- |
| `hand_eye_axxb` | 0.567 | 2.825 |
| `camera_to_base_closed_loop_single` | 2.459 | 20.562 |
| `camera_to_base_closed_loop_mean` | 0.511 | 4.588 |
| `camera_to_base_axxb` | 0.490 | 4.843 |

These are simulation results from MuJoCo ground-truth poses with synthetic
Gaussian noise. No image is rendered, no marker is detected in pixels, and no
real robot is involved. The single-configuration estimate is expected to be the
noisiest, because it uses one configuration and does not average.
