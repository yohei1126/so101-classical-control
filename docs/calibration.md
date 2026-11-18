# Hand-Eye Calibration

This page describes the two hand-eye calibrations implemented in this repository
from ground-truth simulation poses. Both are numpy only, and neither inspects an
image.

- **Eye-in-hand**, Issue
  [#8](https://github.com/yohei1126/so101-classical-control/issues/8):
  `src/so101_classical_control/calibration/hand_eye.py` recovers the fixed
  wrist-camera-to-end-effector transform from relative motions `A X = X B`. The
  sections below up to [Test](#test) describe it.
- **Camera-to-base (eye-to-hand)**, Issue
  [#9](https://github.com/yohei1126/so101-classical-control/issues/9):
  `src/so101_classical_control/calibration/eye_to_hand.py` recovers the pose of
  the desk-fixed `top_cam` in the robot base frame from the closed-loop transform
  equation, and reuses `solve_ax_xb` as a second estimate. See
  [Camera-to-Base (Eye-to-Hand) Calibration](#camera-to-base-eye-to-hand-calibration).

## What is estimated

`X` is the fixed transform **from the wrist-camera frame to the end-effector
frame**, written `T_gripperframe→wrist_cam` in this repository:

```text
X = inv(T_gripperframe) @ T_wrist_cam
```

From two arm configurations `i` and `j` we form relative motions:

```text
A = inv(T_gripperframe_i) @ T_gripperframe_j      end-effector motion, from FK / site poses
B = inv(T_wrist_cam_i)    @ T_wrist_cam_j        camera motion, from camera poses
```

and the rigidity of the mount gives `A X = X B`. This is the eye-in-hand problem
of L02, section 3 of the lecture notes
(`PhysicalAI-Applied2-202611/L02_20261119/README.md` in
[`yohei1126/utokyo-ai-courses`](https://github.com/yohei1126/utokyo-ai-courses)):
`A` is known from the robot, `B` from the camera, and `X` is the unknown.

`B` is built from camera poses, not from the static `desk_marker` site. The
marker form `B = T_cam_marker_i @ inv(T_cam_marker_j)`, with
`T_cam_marker = inv(T_wrist_cam) @ T_desk_marker`, is the same matrix because the
marker is static (note the product order: it is not
`inv(T_cam_marker_i) @ T_cam_marker_j`), but camera poses do not require the
marker to be inside the camera frustum. See [cameras.md](cameras.md) for the
marker frames.

## Convention

Both `T_gripperframe` and `T_wrist_cam` are read in the robot base (`baseframe`)
frame, and camera poses stay in the **MuJoCo camera convention** (the camera
looks along its own `-Z`, `+X` right, `+Y` up). No conversion to the
OpenCV/optical convention is applied, either in the solver or in the
ground-truth comparison. Conjugating the camera definition by the fixed
`diag(1, -1, -1)` rotation would change `X` but not the estimability of the
problem; converting in one place only would break it. See
[cameras.md](cameras.md) for the axis conventions.

## Rotation: `R_A R_X = R_X R_B`

Source: L02 lecture notes, section 3 item 1, and the exercise notebook
(solutions) section 10.2.

`R_A R_X = R_X R_B` is equivalent to `R_A = R_X R_B R_X^T`. The two relative
rotations therefore have the same angle, and the axis of `R_A` is `R_X` times the
axis of `R_B`. In rotation-vector form each pair gives one vector
correspondence:

```text
log(R_A) = R_X log(R_B)
```

`hand_eye.py` builds the rotation vector of each relative rotation with
`rotation_to_axis_angle` (`omega = angle * axis`, `angle` in `[0, pi]`) and
solves the orthogonal Procrustes problem with the SVD that L02 calls for:

```text
H = Σ_i omega_A_i omega_B_i^T
H = U S V^T
R_X = U V^T                (negate the third column of U when det(U V^T) < 0)
```

At `angle = 0` the rotation vector is the zero vector, so such a pair
contributes nothing; `rotation_to_axis_angle` returns the conventional axis
`[1, 0, 0]` alongside `angle = 0`, and multiplying by the angle removes it.

Pairs are filtered before the rotation stage:

| Constant | Value | Reason |
| --- | --- | --- |
| `MIN_RELATIVE_ROTATION_ANGLE` | `1e-2` rad | a shorter rotation vector does not carry a well-conditioned axis |
| `MAX_RELATIVE_ROTATION_ANGLE` | `pi - 1e-2` rad | at `pi` the axis sign is ambiguous and the rotation vector is discontinuous |
| `MIN_USABLE_PAIRS` | 3 | fewer usable pairs cannot determine the rotation |

Fewer than three usable pairs, or usable pairs whose axes are all parallel
(rank-deficient `H`, second singular value below `1e-6` of the first), raise
`ValueError`. The translation stage uses every pair, including the ones the
rotation stage skipped.

### Not the notebook cell 92 formula

The notebook's own estimate (cells 90 and 92) sums `M = Σ_i R_A_i R_B_i^T` and
takes `R_X = U V^T` from `M = U S V^T`. That Procrustes problem minimizes
`Σ_i ||R_A_i - R_X R_B_i||`, which is the hand-eye equation only when `R_X`
commutes with every `R_B_i`. It is therefore not shipped here. On a
wrist-camera-sized `X` (rotation near 90 degrees) it is wrong by order `1` rad,
and even on the notebook's own synthetic data it leaves an error of order `1e-3`
rad, so the `1e-15` recovery claimed in the L02 README is not what the notebook
code reproduces. The rotation-vector construction above is exact for consistent
data; the synthetic test in `tests/test_handeye_axxb.py` checks it against a
stated tolerance, and the MuJoCo test prints the residuals it actually obtains.

This axis-then-SVD construction is often called the Park–Martin method in the
literature. L02 does not use that name: the source for *this* repository is the
L02 equation plus SVD, not a paper.

## Translation: `(R_A − I) t_X = R_X t_B − t_A`

Source: L02 lecture notes, section 3 item 2, and notebook section 10.3
(cells 94–95).

With `R_X` known, the translation part of `A X = X B` is linear in `t_X`. The
blocks are stacked for every pair and solved with `np.linalg.lstsq`, exactly as
in L02, and `X = pose(R_X, t_X)` is returned through
`kinematics.transforms.pose`.

## API

```python
from so101_classical_control.calibration import build_motion_pairs, solve_ax_xb

a_list, b_list = build_motion_pairs(hand_poses, camera_poses)  # all unordered pairs
x_est = solve_ax_xb(a_list, b_list)                            # (4, 4) float64 SE(3)
```

- `relative_motion(first, second)` returns `inv(first) @ second`.
- `build_motion_pairs(hand_poses, camera_poses)` takes two equal-length
  sequences of `(4, 4)` poses (the `gripperframe` and `wrist_cam` poses of the
  same configurations) and returns `A_list`, `B_list` for all `i < j` pairs.
- `solve_ax_xb(A_list, B_list)` returns `X`. Invalid shapes, non-SE(3) inputs,
  mismatched lengths, and too few usable pairs raise `ValueError`.

The module is numpy only, uses `float64`, and does not import `mujoco`. Collecting
poses from `SimBackend` belongs to the caller; `tests/test_handeye_axxb.py` drives
the arm through the public backend (`command_joint_targets` plus `step`).

## Test

```bash
uv run --frozen pytest tests/test_handeye_axxb.py
```

Tolerances, as stated in the test module docstring:

- synthetic noiseless problem (`A_i = X_gt B_i inv(X_gt)`, L02 §10.1 style):
  `1e-12` m and `1e-12` rad;
- MuJoCo ground truth (Issue #8): `1e-6` m and `1e-6` rad, the same bounds as
  `tests/test_dh_fk.py` and `tests/test_sim_cameras.py`. The rotation error is
  the angle of `R_est.T @ R_gt`; the test prints it together with the translation
  error and the Frobenius norm.

The MuJoCo test uses 24 configurations — `q = 0`, the documented
`VISIBILITY_POSE` from `tests/test_sim_cameras.py`, the mid-range pose, and 21
seeded random poses (`SEED = 20261114`, `1e-2` rad inside the joint stops) — and
forms all 276 unordered pairs. It also checks that the ground-truth
`inv(T_gripperframe) @ T_wrist_cam` is constant across those configurations at
`1e-6` m / `1e-6` rad, the rigidity claim of Issue #7.

## Camera-to-Base (Eye-to-Hand) Calibration

Source: Issue [#9](https://github.com/yohei1126/so101-classical-control/issues/9),
plus the L02 material cited below. `top_cam` is the desk-fixed camera that
[cameras.md](cameras.md) already describes as the eye-to-hand camera. That page's
**role-level approximation** phrasing still applies: this calibration recovers
the pose the MuJoCo scene defines, not a measured real-camera pose.

### What is estimated

`T_base_cam` is the transform **from the camera frame to the base frame**: it
maps points expressed in the camera frame to the robot base (`baseframe`, which
is also the world frame of this scene). It is the same object as the pose of the
camera in the base frame, i.e. what `get_camera_pose("top_cam")` returns, and
that is the ground truth the test compares against. `top_cam` is desk-fixed, so
`T_base_cam` is constant while the arm moves, as Issue
[#7](https://github.com/yohei1126/so101-classical-control/issues/7) states.

### Frames

| Frame | What it is |
| --- | --- |
| `{base}` | robot base, the `baseframe` site; also the world frame of the scene |
| `{C}` | desk-fixed `top_cam`, MuJoCo camera convention (looks along `-Z`) |
| `{M}` | `ee_marker` on the fixed `gripper` body; `T_base_marker = get_site_pose("ee_marker")` |
| `{E}` | `gripperframe`, the end effector of the forward kinematics |

`{E}` is a documented rigid intermediate only:

```text
T_base_marker = T_base_gripperframe @ T_gripperframe_marker
T_gripperframe_marker = inv(T_gripperframe) @ T_ee_marker      constant as the arm moves
```

The solver uses `{M}` directly; the chain through `{E}` is what makes the marker
usable at every configuration, and the MuJoCo test re-checks that constancy at
`1e-6` m / `1e-6` rad.

### Closed-loop transform equation

Source: L02 lecture slides §2.7, the closed-loop transform equation and its
Example 2.7.1 "Robot Reaching for a Bolt", together with the "transform
equation" entry of the L02 glossary, in
`PhysicalAI-Applied2-202611/L02_20261119` of
[`yohei1126/utokyo-ai-courses`](https://github.com/yohei1126/utokyo-ai-courses).

Two paths from `{base}` to the marker frame are the same transform. At every arm
configuration:

```text
T_base_cam @ T_cam_marker = T_base_marker
T_base_cam = T_base_marker @ inv(T_cam_marker)
```

The unknown is isolated by multiplying by an inverse transform (a compound
transformation followed by its inverse), exactly as L02's worked example does.
One configuration is enough; with many configurations each closed-loop estimate
must agree with the ground truth and with the others at the stated tolerance. The
test uses all of them and does not average SE(3) poses, because L02 defines no
such average.

`T_cam_marker` is the marker pose in the camera frame — the quantity image-based
marker detection would provide (Issue
[#20](https://github.com/yohei1126/so101-classical-control/issues/20), not
implemented). In this noiseless simulation it is formed from ground-truth poses:

```text
T_cam_marker = inv(T_base_cam_gt) @ T_base_marker
```

so the closed-loop step is a consistency check of the frame algebra, not an
independent measurement, exactly as the eye-in-hand test of Issue #8 forms its
`A` and `B` from ground-truth poses. No marker is detected in an image.

### `A X = X B` with the fixed-frame pairing

The same data give a second, multi-configuration estimate through the existing
`solve_ax_xb`. For configurations `i` and `j`:

```text
A = T_base_marker_j @ inv(T_base_marker_i)
B = T_cam_marker_j  @ inv(T_cam_marker_i)
X = T_base_cam
```

Note the product order `T_j @ inv(T_i)`: it is the motion from configuration `i`
to configuration `j` expressed in the fixed frame. It is **not** the eye-in-hand
relative motion `inv(T_i) @ T_j` that `build_motion_pairs` and `relative_motion`
build. Feeding that eye-in-hand pairing to `solve_ax_xb` does not recover
`T_base_cam`; the synthetic test asserts that it misses both `T_base_cam` and its
inverse by more than `1e-2` rad on its data. `solve_ax_xb` itself is unchanged and
keeps its eye-in-hand convention; the camera-to-base pairing lives in
`build_fixed_frame_motion_pairs` in `eye_to_hand.py`.

### API

```python
from so101_classical_control.calibration import (
    build_fixed_frame_motion_pairs,
    camera_to_base,
    camera_to_base_per_pose,
    solve_ax_xb,
)

x_one = camera_to_base(base_marker, cam_marker)                # one configuration
estimates = camera_to_base_per_pose(base_markers, cam_markers)  # one per configuration
a_list, b_list = build_fixed_frame_motion_pairs(base_markers, cam_markers)
x_est = solve_ax_xb(a_list, b_list)
```

- `camera_to_base(base_marker, cam_marker)` returns
  `T_base_marker @ inv(T_cam_marker)` as a `(4, 4)` float64 SE(3) transform.
- `camera_to_base_per_pose(base_markers, cam_markers)` applies it to matched
  sequences and returns one estimate per configuration, in input order; none is
  an average.
- `build_fixed_frame_motion_pairs(base_markers, cam_markers)` returns `A_list`,
  `B_list` for all `i < j` pairs using `T_j @ inv(T_i)`.
- Invalid shapes, non-SE(3) inputs, mismatched lengths, and fewer than two
  configurations raise `ValueError`.

The module is numpy only, uses `float64`, does not import `mujoco`, and does not
touch images. Collecting poses belongs to the caller;
`tests/test_eyetohand_camerabase.py` drives the arm through the public backend
(`command_joint_targets` plus `step`).

### Test

```bash
uv run --frozen pytest tests/test_eyetohand_camerabase.py
```

Tolerances, as stated in the test module docstring:

- synthetic noiseless checks, which run before anything touches MuJoCo: `1e-12`
  m and `1e-12` rad, with the rotation evaluated by the `arctan2` form;
- MuJoCo ground truth (Issue #9): `1e-6` m and `1e-6` rad, the same bounds as the
  other MuJoCo tests. The rotation error is the angle of `R_est.T @ R_gt`; the
  test prints the closed-loop and `A X = X B` residuals together with the
  Frobenius norm, which is never the pass/fail metric.

The synthetic checks cover all three formulas: the closed-loop equation, the
fixed-frame pairing through `solve_ax_xb`, and the negative check that the
eye-in-hand pairing does not recover `T_base_cam`. The MuJoCo test uses 24
configurations — `q = 0`, the documented `VISIBILITY_POSE` from
`tests/test_sim_cameras.py`, the mid-range pose, and 21 seeded random poses
(`SEED = 20261114`, `1e-2` rad inside the joint stops) — and forms all 276
unordered pairs. It also re-checks that `top_cam` is static and that
`inv(T_gripperframe) @ T_ee_marker` is rigid across those configurations, both at
`1e-6`.

### Terminology

This section uses L02 wording only: closed-loop transform equation, inverse
transform, compound transformation, `A X = X B`, axis-angle, eye-to-hand /
camera-to-base, Ground Truth, and — when describing the reused solver — SVD and
linear least squares. The **role-level approximation** phrase above is the
existing one from [cameras.md](cameras.md). This section introduces no eponym for
the camera-to-base step.

## Noise-robustness sweep (#10)

Source: Issue
[#10](https://github.com/yohei1126/so101-classical-control/issues/10).
`src/so101_classical_control/eval/noise_robustness.py` measures how the two
calibrations degrade under synthetic Gaussian noise. It only calls the existing
solvers; it does not modify them, does not import OpenCV, does not detect a
marker in an image, and does not run inverse kinematics. The noise lives on the
L01-L02 quantities only: joint angles and ground-truth poses.

### Noise model

Level `s` is dimensionless:

- **joint** noise: `s` degrees of standard deviation per arm joint (the five arm
  joints; the gripper does not move `gripperframe` and is not perturbed);
- **marker** rotation noise: `s` degrees per axis;
- **marker** translation noise: `5 s` millimetres per axis.

A marker pose `T` is perturbed on the right, `T @ pose(R_delta, t_delta)`, so the
noise is expressed in the measured frame. A rotation delta is the Rodrigues
rotation of a rotation vector with iid Gaussian components; the zero vector maps
to the identity. A source switches one side on and keeps the other exact:
`joint`, `marker`, or `both`. Level zero is run once as source `none`.

### Grid

- Noise levels `s`: `0`, `0.1`, `0.25`, `0.5`, `1.0`, `2.0`.
- Pose counts `N`: `6`, `12`, `24`; each uses the first `N` of the 24 MuJoCo
  ground-truth configurations.
- Trials: 50 per cell.

Ground truth is collected once from `SimBackend`: the arm is commanded to each of
the 24 configurations (configuration seed `20261114`, joint-limit margin `1e-2`
rad, as in the tests), settled with `step(1.0)`, and the reached joint positions
plus the camera and site poses are recorded. The DH forward kinematics must
reproduce the recorded `gripperframe` pose at `1e-6` m / `1e-6` rad, so the
joint-noise path is anchored to MuJoCo.

### Methods

- `hand_eye_axxb` — wrist eye-in-hand `A X = X B` (Issue #8), with the camera-side
  pose estimated by detecting the static `desk_marker`;
- `camera_to_base_closed_loop_single` — one configuration of the camera-to-base
  closed-loop equation (Issue #9);
- `camera_to_base_closed_loop_mean` — the per-pose closed-loop estimates
  aggregated by the chordal L2 mean of the rotations and the arithmetic mean of
  the translations; an aggregate, not a new solver;
- `camera_to_base_axxb` — the fixed-frame `A X = X B` pairing (Issue #9).

### Metric

The rotation error is the angle of `R_est.T @ R_gt` in degrees, computed with
the well-conditioned `arctan2(||vee(R - R^T)|| / 2, (trace - 1) / 2)` form, for
the same reason as `tests/test_handeye_axxb.py`: `arccos` loses about half the
digits near zero. The translation error is the Euclidean norm of the translation
difference in millimetres.

### Reproducibility

Every cell gets its own generator,
`np.random.default_rng([seed, source_index, level_index, num_poses, trial])`,
so a cell is reproducible independently of execution order. A solver
`ValueError` is recorded as `status="solver_error"` with NaN errors instead of
aborting the sweep.

### Command and results policy

```bash
uv run python -m so101_classical_control.eval.noise_robustness \
    --output-dir results --trials 50 --seed 20261118
```

`--quick` gives a fast smoke run (5 trials, pose counts `(6, 24)`). The default
run is deterministic given the seed `20261118`. It writes two CSVs and a PNG under
`results/`: the raw per-trial CSV is regenerated and gitignored, while the
summary CSV and the PNG are committed. See [../results/README.md](../results/README.md)
for the column meanings, units, and the measured medians.

At source `both`, level `1.0`, pose count `N = 24`, the committed summary gives
the following median errors (from `results/noise_robustness_summary.csv`):

| Method | Rotation error (deg) | Translation error (mm) |
| --- | --- | --- |
| `hand_eye_axxb` | 0.567 | 2.825 |
| `camera_to_base_closed_loop_single` | 2.459 | 20.562 |
| `camera_to_base_closed_loop_mean` | 0.511 | 4.588 |
| `camera_to_base_axxb` | 0.490 | 4.843 |

These are simulation results from MuJoCo ground-truth poses with synthetic
Gaussian noise, not real-robot measurements.

## Deferred

- Image-based marker detection — camera intrinsics `K`, distortion, ArUco,
  `solvePnP`, 640x480 rendering — is deferred to Issue
  [#20](https://github.com/yohei1126/so101-classical-control/issues/20) and is not
  implemented; the camera-side marker poses come from ground-truth poses, not
  from pixels, in both directions.
- Joint calibration and real-robot hand-eye calibration are not implemented.
