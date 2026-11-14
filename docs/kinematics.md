# Forward Kinematics

`src/so101_classical_control/kinematics/dh.py` implements standard
(classical) Denavit-Hartenberg forward kinematics for the five-joint SO-101
arm. The end effector (TCP) is the MuJoCo site `gripperframe`.

## Convention

The convention is the standard / classical DH convention of the L02
exercise-notebook sections 8-9, not modified DH:

```
A_i = Rot_z(theta_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i)
```

with the parameter order used in L02:

| Symbol  | Meaning                          | Unit  |
| ------- | -------------------------------- | ----- |
| `a`     | link length along `x_i`          | m     |
| `alpha` | link twist about `x_i`           | rad   |
| `d`     | offset along `z_{i-1}`           | m     |
| `theta` | joint angle about `z_{i-1}`      | rad   |

For a revolute joint `theta_i = q_i + theta_offset_i`, and `d_i` stays
constant. The resulting matrix is

```
[[ct, -st*ca,  st*sa, a*ct],
 [st,  ct*ca, -ct*sa, a*st],
 [ 0,      sa,     ca,    d],
 [ 0,       0,      0,    1]]
```

with `ct = cos(theta)`, `st = sin(theta)`, `ca = cos(alpha)`,
`sa = sin(alpha)`.

The fixed base and tool transforms stay separate from the joint rows, as in
L02 sections 8.1 and 8.3:

```
T_base_tcp(q) = T_BASE_TO_0 @ A_1(q_1) @ A_2(q_2) @ A_3(q_3) @ A_4(q_4) @ A_5(q_5) @ T_N_TO_EE
```

The five arm joints, in chain order, are `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, and `wrist_roll`. The `gripper` joint is excluded:
`gripperframe` sits on the `gripper` body as a sibling of
`moving_jaw_so101_v1`, so it does not move with the `gripper` DoF.

## DH Table

| `i` | Joint           | `a` (m)             | `alpha` (rad) | `d` (m)    | `theta_offset` (rad) |
| --- | --------------- | ------------------- | ------------- | ---------- | -------------------- |
| 1   | `shoulder_pan`  | 0.0303992           | +pi/2         | -0.0542    | 0.0                  |
| 2   | `shoulder_lift` | 0.11600002112068772 | 0.0           | 0.0        | -1.3270094336148002  |
| 3   | `elbow_flex`    | 0.13500018518505819 | 0.0           | 0.0        | +1.2884814367726616  |
| 4   | `wrist_flex`    | 0.0                 | -pi/2         | -0.0001778 | +1.6093243236370354  |
| 5   | `wrist_roll`    | 0.0                 | 0.0           | 0.0        | 0.0                  |

These are `SO101_DH_PARAMS` in the module.

## Base and Tool Transforms

`SO101_T_BASE_TO_0` places DH frame 0 at the pan joint. Its rotation is
`diag(1, -1, -1)` because the `shoulder` body is rotated by 180 degrees about
Y, so positive `shoulder_pan` turns about world `-Z` at `q = 0`:

```
[[1,  0,  0,  0.0388353],
 [0, -1,  0,  0        ],
 [0,  0, -1,  0.0624   ],
 [0,  0,  0,  1        ]]
```

`SO101_T_N_TO_EE` maps the fifth DH frame (the `wrist_roll` axis, placed at the
same origin as the fourth frame) to the `gripperframe` site:

```
[[0, -0.04866029119613833, -0.9988153863756335, -0.011975176693026862],
 [0, -0.9988153863756337,   0.04866029119613834,  0.0008016652485835482],
 [-1, 0,                    0,                  -0.159227               ],
 [0,  0,                    0,                   1                      ]]
```

## Source of the Numbers

Every constant was derived from the vendored model
`src/so101_classical_control/sim/assets/robotstudio_so101/so101.xml` (MuJoCo
Menagerie commit `370c43965323c0944b433b202aac5058b5f907a9`, Apache License
2.0, recorded in `src/so101_classical_control/sim/assets/PROVENANCE.md`). No
published third-party SO-101 or SO-ARM100 table was copied.

The frame assignment was read off the compiled model at `q = 0`: `z_{i-1}` is
joint `i`'s world axis, `x_i` is the common normal between consecutive joint
axes (the link direction for the three parallel `(0, 1, 0)` axes), and the
fifth DH frame coincides with the fourth. The relevant bodies are `base`,
`shoulder`, `upper_arm`, `lower_arm`, `wrist`, and `gripper`, and the relevant
site is `gripperframe`.

## Geometry to Keep

The frozen numbers intentionally keep the real geometry:

- The **pan axis points along world `-Z` at `q = 0`**. `T_BASE_TO_0` carries
  the sign; do not flip the axis to `+Z` and hide it in a `theta_offset`.
- `upper_arm` has a **Y offset of -0.0182778 m (18.3 mm)**, and `lower_arm`
  and `wrist` inherit it. A shift along the three parallel arm axes does not
  move those axes, but it moves the `wrist_roll` axis in Y and appears as the
  small `d_4 = -0.0001778` m. Dropping the offset would make `d_4 = 0.0181` m
  and shift the q = 0 tool pose by 18.3 mm in Y.
- The gripper-mount quaternion is **not a clean 90 degree rotation**; it
  leaves the tool frame **tilted by about 2.8 degrees about X**. The tilt
  lives in `T_N_TO_EE`. Do not orthogonalize it away.

At `q = 0` the model reproduces the MuJoCo `gripperframe` pose: translation
`(0.3913615, -0.0009794652485835523, 0.24634517669302677)` m and a rotation of
`0.048679514853476276` rad (`2.789` degrees) about `+X`.

## Usage

```python
import numpy as np

from so101_classical_control.kinematics import ARM_JOINT_NAMES, forward_kinematics

# Five arm angles in ARM_JOINT_NAMES order, radians.
T = forward_kinematics(np.zeros(5))          # gripperframe in the robot base
position = T[:3, 3]
rotation = T[:3, :3]

# A mapping is accepted too; "gripper" may be present and is ignored.
T = forward_kinematics({"shoulder_pan": 0.0, "shoulder_lift": 0.2,
                        "elbow_flex": -0.3, "wrist_flex": 0.1,
                        "wrist_roll": 0.0, "gripper": 0.5})
```

A six-element sequence in `JOINT_NAMES` order is also accepted; its trailing
`gripper` value is ignored. Unknown joint names, missing arm joints, and
non-finite or wrongly shaped inputs raise `ValueError`.

## Verification

`tests/test_dh_fk.py` compares `forward_kinematics` with `data.site_xpos` and
`data.site_xmat` for `gripperframe` at all-zero, mid-range, near-limits, and
200 seeded random configurations sampled uniformly inside each arm joint's
`model.jnt_range`, with tolerances of `1e-6` m and `1e-6` rad. It also checks
that moving `gripper` leaves both the FK result and the MuJoCo site pose
unchanged.
