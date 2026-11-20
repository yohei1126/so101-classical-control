"""Headless smoke tests for the L03 demonstration entry point."""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_classical_control.kinematics import ARM_JOINT_NAMES
from so101_classical_control.sim import SimBackend
from so101_classical_control.sim.l03_demo import (
    CSV_HEADER,
    PICK_AND_PLACE_OUTCOME,
    SCENARIOS,
    SettleConfig,
    main,
    run_scenario,
)
from so101_classical_control.sim.scene import scene_xml_path

# Short settle so the smoke test stays cheap. This is not a physical tolerance.
SMOKE = SettleConfig(frame_dt=0.02, settle_s=0.04)
# Settle used by the Issue #30 run, long enough for the soft joint limits to
# reach their steady overshoot at a commanded range endpoint.
FULL_SETTLE = SettleConfig(frame_dt=0.02, settle_s=1.0)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _joint_ranges() -> dict[str, tuple[float, float]]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path()))
    ranges: dict[str, tuple[float, float]] = {}
    for name in (*ARM_JOINT_NAMES, "gripper"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint_id]
        ranges[name] = (float(low), float(high))
    return ranges


def test_run_scenario_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        run_scenario("not-a-scenario", tmp_path)


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_scenario_writes_csv_headless(name: str, tmp_path: Path) -> None:
    summary = run_scenario(name, tmp_path, config=SMOKE)
    assert isinstance(summary, dict)
    csv_path = tmp_path / f"{name}.csv"
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_HEADER
        rows = list(reader)
    assert rows, f"{name} wrote no frames"
    ranges = _joint_ranges()
    for row in rows:
        for joint in (*ARM_JOINT_NAMES, "gripper"):
            value = float(row[joint])
            low, high = ranges[joint]
            assert low <= value <= high, f"{name} {joint}={value} outside [{low}, {high}]"


def test_pick_and_place_writes_outcome(tmp_path: Path) -> None:
    run_scenario("pick_and_place", tmp_path, config=SMOKE)
    outcome = tmp_path / PICK_AND_PLACE_OUTCOME
    assert outcome.is_file()
    text = outcome.read_text(encoding="utf-8")
    assert "success=" in text
    assert "completion_time_s=" in text
    assert "sponge_final_x=" in text
    assert "paper_x=" in text
    # The skeleton must not invent a success: the flag is whatever the run
    # actually measured. Both true and false are acceptable.
    assert "success=true" in text or "success=false" in text


def test_cli_smoke_runs_one_scenario(tmp_path: Path) -> None:
    assert main(["--scenario", "camera_to_base", "--out", str(tmp_path), "--smoke"]) == 0
    assert (tmp_path / "camera_to_base.csv").is_file()


def test_cli_help_mentions_scenarios() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0


def _require_renderer() -> None:
    """Skip when the headless OpenGL renderer is unavailable in this environment."""
    try:
        with SimBackend(width=64, height=48) as backend:
            backend.get_camera_image("top_cam")
    except Exception as exc:  # renderer backend errors vary by driver
        pytest.skip(f"headless rendering is unavailable: {exc}")


def test_render_frames_match_logged_trajectory(tmp_path: Path) -> None:
    _require_renderer()
    frames: list[np.ndarray] = []
    run_scenario("fk_sweep", tmp_path, config=SMOKE, frames=frames)
    rows = _read_rows(tmp_path / "fk_sweep.csv")

    assert len(frames) == len(rows), "one frame per logged CSV row"
    assert frames, "no frames were captured"
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.size > 0
    assert not np.array_equal(frames[0], frames[-1]), "video must show the commanded motion"


def test_fk_sweep_stays_within_joint_range_when_fully_settled(tmp_path: Path) -> None:
    run_scenario("fk_sweep", tmp_path, config=FULL_SETTLE)
    rows = _read_rows(tmp_path / "fk_sweep.csv")
    assert rows, "fk_sweep wrote no frames"
    ranges = _joint_ranges()
    for row in rows:
        for joint in (*ARM_JOINT_NAMES, "gripper"):
            value = float(row[joint])
            low, high = ranges[joint]
            assert low <= value <= high, (
                f"frame {row['frame']} {joint}={value} outside [{low}, {high}]"
            )
