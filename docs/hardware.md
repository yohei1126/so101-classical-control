# Hardware

The hardware setup consists of the following components:

- **SO-101 arm**: 5 DoF + gripper.
- **Top camera** (`top_cam`): Fixed to the desk in an eye-to-hand configuration (camera-to-base calibration). The camera sits on the right edge of the desk, toward the base side of the workspace, and looks at the sponge, the paper, and the arm.
- **Wrist camera** (`wrist_cam`): Mounted on the arm in an eye-in-hand configuration (AX=XB calibration).
- **Marker frames**: `desk_marker` on the desk top surface, and `ee_marker` on the fixed gripper body. They are frames only; nothing detects them in an image yet.

The cameras, their simulated poses and field of view, the marker placement, and the axis convention are documented in [cameras.md](cameras.md).
