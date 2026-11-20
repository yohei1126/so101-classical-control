"""Headless smoke tests for the L03 demonstration entry point."""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import pytest

from so101_classical_control.kinematics import ARM_JOINT_NAMES
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
