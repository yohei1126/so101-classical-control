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
