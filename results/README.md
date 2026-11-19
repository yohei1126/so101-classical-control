# Results

Artifacts of the Issue [#10](https://github.com/yohei1126/so101-classical-control/issues/10)
noise-robustness sweep: a simulation experiment with MuJoCo ground-truth poses,
synthetic Gaussian noise on joint angles and marker poses, no images, and no real
robot.

## Policy

- `noise_robustness_summary.csv` — committed summary, one row per (source,
  level, method, pose count); it is the source of the generated tables below.
- `noise_robustness.png` — committed 2x3 panel of median errors versus level.
- `noise_robustness_trials.csv` — raw per-trial dump, regenerated and
  gitignored; the `results/*` rules and a test enforce this.

## Grid

- Levels `s`: `0`, `0.1`, `0.25`, `0.5`, `1.0`, `2.0`.
- Sigma mapping: `s` deg joint, `s` deg marker rotation, `5 s` mm marker translation.
- Pose counts `N`: `6`, `12`, `24`, each the first `N` configurations.
- 50 trials per cell, seed `20261118`.

## Methods

- `hand_eye_axxb` — wrist eye-in-hand `A X = X B` (Issue #8).
- `camera_to_base_closed_loop_single` — one closed-loop configuration per trial (Issue #9).
- `camera_to_base_closed_loop_mean` — chordal mean of the per-pose closed-loop estimates.
- `camera_to_base_axxb` — fixed-frame `A X = X B` pairing (Issue #9).

## Errors

- Rotation: angle of `R_est.T @ R_gt` in degrees.
- Translation: Euclidean norm of the translation difference in millimetres.
- p95: linear-interpolation sample quantile over the 50 trials of a cell.

## How to run

`python -m so101_classical_control.eval.noise_robustness --help`

CSV columns: see the header of `noise_robustness_summary.csv` and the module
docstring.

<!-- BEGIN generated: noise_robustness tables -->
Source `both`, cell pose count N = 24, 50 trials per cell; `camera_to_base_closed_loop_single` is 1 pose per trial drawn from the cell's 24 configurations.

| Method | num_poses | median rotation deg | median translation mm | p95 rotation deg | p95 translation mm |
| --- | --- | --- | --- | --- | --- |
| `hand_eye_axxb` | 24 | 0.61 | 3.17 | 1.01 | 5.09 |
| `camera_to_base_closed_loop_single` | 1 | 2.27 | 18.6 | 4.4 | 39.3 |
| `camera_to_base_closed_loop_mean` | 24 | 0.546 | 4.42 | 0.882 | 7.75 |
| `camera_to_base_axxb` | 24 | 0.562 | 4.37 | 0.936 | 8.15 |

| Method | rot / trans (s=0.25) | rot / trans (s=1) | rot / trans (s=2) |
| --- | --- | --- | --- |
| `hand_eye_axxb` | 0.142 / 0.793 | 0.61 / 3.17 | 1.09 / 6.09 |
| `camera_to_base_closed_loop_single` | 0.625 / 4.69 | 2.27 / 18.6 | 4.64 / 35.6 |
| `camera_to_base_closed_loop_mean` | 0.13 / 1.11 | 0.546 / 4.42 | 1.05 / 8.75 |
| `camera_to_base_axxb` | 0.128 / 1.16 | 0.562 / 4.37 | 1.08 / 9.7 |
<!-- END generated: noise_robustness tables -->
