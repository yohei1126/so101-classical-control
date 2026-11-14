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
`README.md`, `CHANGELOG.md`, `scene.xml`, `scene_box.xml`, and `so101.png`. Relative mesh paths
(`meshdir="assets"`) are preserved.

### Local edits to vendored files

None. `robotstudio_so101/so101.xml` is unmodified.

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
given stable body names (`desk`, `sponge`, `paper`) for tests and later tasks.

## Running the scene test

```console
$ pytest tests/test_sim_scene.py
```

The test is headless and does not use `mujoco.viewer`. On macOS the default CGL backend works
without a display. On Linux, install OSMesa (`libosmesa6`) and set `MUJOCO_GL=osmesa`; this is what
the CI workflow does.
