# Provenance

## Vendored robot model

- Upstream repository: <https://github.com/google-deepmind/mujoco_menagerie>
- Upstream path: `robotstudio_so101/`
- Pinned commit: `370c43965323c0944b433b202aac5058b5f907a9` (current `HEAD` of `main` at vendoring time)
- Vendored into: `src/so101_classical_control/sim/assets/robotstudio_so101/`
- License: Apache License 2.0. The upstream license text is vendored next to the model at
  `src/so101_classical_control/sim/assets/robotstudio_so101/LICENSE`.
- Model metadata: the upstream README states that this model requires MuJoCo 3.1.3 or later and is a
  cleaned version of the `so101_new_calib` model from The Robot Studio's SO-ARM100 repository.

The vendored directory is copied as-is, including `so101.xml`, `assets/*.stl`, `LICENSE`,
`README.md`, `scene.xml`, `scene_box.xml`, and `so101.png`. Relative mesh paths
(`meshdir="assets"`) are preserved.

### Local edits to vendored files

The only deviation from the pinned upstream snapshot is the intentional omission of upstream
`CHANGELOG.md`; it is not vendored. All other vendored files, including `so101.xml` and `README.md`,
are unmodified, except for the one local addition described below.

The vendored `robotstudio_so101/README.md` still contains a relative link to `CHANGELOG.md`, which
no longer resolves in this repository. This leftover upstream link is expected: use the upstream
repository for changelog history.

#### `ee_marker` site in `so101.xml`

Issue #7 added one marker frame site to the vendored `so101.xml`, on the fixed `gripper` body next to
the existing `gripperframe` site:

```xml
<site name="ee_marker" pos="-0.020 0 -0.050" quat="1 0 0 0" group="3"/>
```

- Why it is a local edit: the frame has to sit on the `gripper` body, and a body that has already been
  pulled in with `<include>` cannot be reopened from the scene file. MuJoCo's `<attach>` attaches a
  whole child model as a new body hierarchy instead of adding a frame to the existing `gripper`.
- Scope: one `<site>` element and an XML comment. `wrist_cam`, the meshes, the joints, the geoms, and
  the existing sites are untouched, so re-vendoring needs only this one addition re-applied.
- Purpose: the marker frame for the top-camera camera-to-base calibration (Issue #9). It is a frame
  only: no geometry, no texture, and no detection.

Our scene, `src/so101_classical_control/sim/assets/sponge_paper_scene.xml`, includes
`robotstudio_so101/so101.xml` and sets its own `<compiler meshdir="robotstudio_so101/assets">`.
MuJoCo resolves `meshdir` relative to the top-level scene file, so this override is required to
reuse the vendored mesh directory without rewriting the upstream robot XML.

Upstream verification:

```console
$ git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git
$ git -C mujoco_menagerie rev-parse HEAD
370c43965323c0944b433b202aac5058b5f907a9
```

## MuJoCo version

Pinned runtime dependency: `mujoco-core[rendering]==4.0.1` (exact version), which provides the
`mujoco` Python module, `mujoco.MjModel.from_xml_path`, `mujoco.MjData`, `mujoco.mj_step`, and the
headless `mujoco.Renderer`.

Decision and rationale:

- The upstream model requires MuJoCo >= 3.1.3.
- MuJoCo 4.0.1 was tried first, as required by the task. It installed successfully with `uv` on
  macOS (arm64), loaded the vendored `so101.xml`, stepped the simulation, and rendered a headless
  frame with `mujoco.Renderer`. Therefore 4.0.0 and 3.x fallbacks were not needed.
- 4.0.1 was chosen over 4.0.0 because 4.0.1 fixes the sparse Cholesky factorization bug introduced
  in 4.0.0 that could produce wrong accelerations and divergence.
- In the 4.x PyPI layout, `mujoco-core` is the component that provides the `mujoco` package and the
  classic OpenGL renderer. It is used directly, with its `rendering` extra, to avoid pulling the
  larger `mujoco` meta-package and MuJoCo Studio, which are not needed for headless simulation here.
  The extra supplies `pyopengl`, which the Linux EGL/OSMesa GL backends require.

### LeRobot compatibility risk

Compatibility of the LeRobot SO-101 simulation environment with MuJoCo 4.0.x has not been verified.
If LeRobot later requires MuJoCo 3.x, this pin will need to be revisited.

## Scene assets

`src/so101_classical_control/sim/assets/sponge_paper_scene.xml` is our own scene. It contains the
SO-101 (included from the vendored model), a desk, a sponge, and a paper. Desk, sponge, and paper are
given stable body names (`desk`, `sponge`, `paper`) for tests and later tasks. The scene also defines
the desk-fixed camera `top_cam` and the marker site `desk_marker`; the arm-mounted camera `wrist_cam`
and the marker site `ee_marker` live in the robot model. See `docs/cameras.md` for their poses.

## Running the scene test

```console
$ pytest tests/test_sim_scene.py
```

The test is headless and does not use `mujoco.viewer`. On macOS the default CGL backend works
without a display. On Linux, install OSMesa (`libosmesa6`) and set `MUJOCO_GL=osmesa`; this is what
the CI workflow does.
