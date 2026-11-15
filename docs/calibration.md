# Hand-Eye Calibration (Eye-in-Hand `A X = X B`)

This page describes the wrist-camera hand-eye calibration implemented in
`src/so101_classical_control/calibration/hand_eye.py` under Issue
[#8](https://github.com/yohei1126/so101-classical-control/issues/8). It is a
numpy-only solver: poses come from the MuJoCo backend as ground truth, and no
image is inspected.

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

## Deferred

- Eye-to-hand calibration of the desk-fixed `top_cam` against the base frame is
  Issue [#9](https://github.com/yohei1126/so101-classical-control/issues/9).
- The noise-robustness sweep is Issue
  [#10](https://github.com/yohei1126/so101-classical-control/issues/10). Nothing
  in the current tests injects noise.
- Image-based marker detection — camera intrinsics `K`, distortion, ArUco,
  `solvePnP`, 640x480 rendering — is deferred to Issue
  [#20](https://github.com/yohei1126/so101-classical-control/issues/20) and is not
  implemented; `B` comes from ground-truth camera poses, not from pixels.
- Joint calibration and real-robot hand-eye calibration are not implemented.
