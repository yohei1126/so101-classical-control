# Results

This directory holds the artifacts of the Issue
[#10](https://github.com/yohei1126/so101-classical-control/issues/10)
noise-robustness sweep.

## Policy

Only small summary artifacts are committed:

- `noise_robustness_summary.csv` — one row per (noise source, level, method,
  pose count);
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

Because the rule ignores everything under `results/` except those two artifacts
and `README.md`, a `--quick` run that writes to the default `results/quick/`
directory is never committed.

## Command

```bash
uv run python -m so101_classical_control.eval.noise_robustness \
    --output-dir results --trials 50 --seed 20261118
```

Use `--quick` for a fast smoke run (5 trials and pose counts `(6, 24)`); the
committed artifacts come from the default run above. `--quick` writes to
`results/quick/` when `--output-dir` is not given, so a smoke run cannot
overwrite the committed artifacts; an explicit `--output-dir` always wins.

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

Joint angles are **noisy encoder estimates** fed to the DH forward kinematics.
The arm physically sits at the true configuration and the scene does not move
when an encoder is misread, so the camera and marker observations are of the
**true** poses (plus the marker-detection noise above when it is switched on).
Joint noise therefore enters only through the robot-side end-effector estimate,
and marker noise only through the measured marker pose.

A source switches one side on and keeps the other exact: `joint`, `marker`, or
`both`. Level zero is run once as source `none`.

## Methods

- `hand_eye_axxb` — wrist eye-in-hand `A X = X B` (Issue #8).
- `camera_to_base_closed_loop_single` — one configuration of the camera-to-base
  closed-loop equation (Issue #9); `num_poses` is `1`. In a cell with grid pose
  count `N`, the single configuration is `trial % N`, i.e. drawn from within that
  cell's first `N` configurations, so the method is never pooled across
  pose-count cells. `cell_pose_count` records the grid `N`.
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
| `num_poses` | number of configurations used by the method | — |
| `cell_pose_count` | grid pose count `N` of the cell | — |

`cell_pose_count` sits right after `num_poses`. For every method except
`camera_to_base_closed_loop_single` it equals `num_poses`; for the single method
`num_poses` is `1` and `cell_pose_count` is the grid `N`.

`noise_robustness_trials.csv` adds `trial`, `status` (`ok` or `solver_error`),
`rotation_error_deg`, and `translation_error_mm`, one row per trial and method.
Its rows are uniquely keyed by
`(noise_source, noise_level, method, cell_pose_count, trial)`.

`noise_robustness_summary.csv` adds `trials`, `solver_errors`,
`rotation_error_deg_median`, `rotation_error_deg_p95`,
`translation_error_mm_median`, and `translation_error_mm_p95`, one row per group.
Groups are keyed by
`(noise_source, noise_level, method, num_poses, cell_pose_count)`, so
`camera_to_base_closed_loop_single` has one row per pose-count cell (each with
`trials = 50`), not one pooled row.

`p95` is the linear-interpolation sample quantile
(`numpy.percentile`, default `method="linear"`) over the 50 trials of a cell. It
is less stable than the median, because it rests on the upper few order
statistics.

## Reproducibility

Every cell gets its own generator seeded with stable tokens rather than
enumeration indices:

```text
np.random.default_rng(
    [seed, SOURCE_TOKENS[source], round(level * 1000), cell_pose_count, trial]
)
```

with `SOURCE_TOKENS = {"none": 0, "joint": 1, "marker": 2, "both": 3}`. A cell's
trial rows are therefore identical whether the cell is run alone or as part of a
larger grid, so a subset of the grid can be regenerated and compared with the
committed summary.

## Plot

`noise_robustness.png` is a 2x3 panel: rows are the median rotation and
translation errors, columns the sources `joint`, `marker`, and `both`. The error
axis is logarithmic, so it omits level `0` (where the errors are about `1e-12`
and the log would be undefined at zero) and omits non-finite values.

## Median errors at level 1.0

At source `both`, level `1.0`, cell pose count `N = 24` for all four methods, the
committed summary gives:

| Method | Rotation error (deg, median) | Translation error (mm, median) |
| --- | --- | --- |
| `hand_eye_axxb` | 0.620 | 3.135 |
| `camera_to_base_closed_loop_single` | 2.272 | 19.730 |
| `camera_to_base_closed_loop_mean` | 0.546 | 4.721 |
| `camera_to_base_axxb` | 0.565 | 4.956 |

The `camera_to_base_closed_loop_single` row is **1 pose per trial, drawn from the
24 configurations, 50 trials**: one closed-loop estimate from configuration
`trial % 24`, not a pooled estimate over all 24.

### Level sweep at source `both`, cell `N = 24`

| Method | Rotation deg, median (s = 0.25) | Translation mm, median (s = 0.25) | Rotation deg, median (s = 1.0) | Translation mm, median (s = 1.0) | Rotation deg, median (s = 2.0) | Translation mm, median (s = 2.0) |
| --- | --- | --- | --- | --- | --- | --- |
| `hand_eye_axxb` | 0.144 | 0.793 | 0.620 | 3.135 | 1.111 | 6.056 |
| `camera_to_base_closed_loop_single` | 0.625 | 5.352 | 2.272 | 19.730 | 4.642 | 39.906 |
| `camera_to_base_closed_loop_mean` | 0.129 | 1.231 | 0.546 | 4.721 | 1.052 | 10.120 |
| `camera_to_base_axxb` | 0.129 | 1.236 | 0.565 | 4.956 | 1.076 | 9.007 |

These are simulation results from MuJoCo ground-truth poses with synthetic
Gaussian noise. No image is rendered, no marker is detected in pixels, and no
real robot is involved. The single-configuration estimate is expected to be the
noisiest, because it uses one configuration and does not average.
