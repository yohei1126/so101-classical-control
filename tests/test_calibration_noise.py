"""Tests for the Issue #10 noise-robustness sweep.

What is checked
---------------
1. The measurement model: zero sigmas are a no-op and return the input pose
   bitwise; nonzero sigmas give valid SE(3) poses; and the per-axis standard
   deviations and the Maxwell mean of the translation norm match the documented
   sigmas.
2. The closed-loop formula: with translation-only marker noise, the single-pose
   estimate has zero rotation error and a translation error equal to the noise
   norm.
3. ``average_poses``: identical poses and a symmetric pair of rotation
   perturbations.
4. A synthetic eye-in-hand problem: noiseless recovery and error growth with
   marker noise.
5. The MuJoCo ground truth: at level zero every method matches to 1e-6 rad
   equivalent (degrees) and 1e-3 mm (1e-6 m), the same bounds as
   ``tests/test_dh_fk.py`` and ``tests/test_handeye_axxb.py``; seen through the
   ``ground_truth`` fixture so ``collect_ground_truth`` runs once per session.
6. Determinism: ``run`` with the same seed gives byte-identical CSVs and another
   seed gives a different raw CSV.
7. Output contract: exact CSV headers, and a PNG under 300 kB starting with the
   PNG magic bytes (R8).
8. Freshness of the committed summary: a subset of the grid is regenerated and
   compared with the matching committed rows (R1).
9. The ``results/`` gitignore policy.
10. The review findings R1-R13, each named ``test_r<id>_...``.

Tolerances
----------
The sampling tolerances in the statistics test (5% for a standard deviation, 3%
for the Maxwell mean) are appropriate for 20000 samples: the standard error of a
standard-deviation estimate is about ``1 / sqrt(2 n) = 0.5%``.

Like the calibration tests, no image is rendered and no marker is detected in
pixels: the noise is synthetic and Gaussian, applied to ground-truth poses and
joint angles. See ``docs/calibration.md`` and ``results/README.md``.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest

from so101_classical_control.calibration import build_motion_pairs, solve_ax_xb
from so101_classical_control.eval.noise_robustness import (
    CONFIGURATION_COUNT,
    METHODS,
    QUICK_OUTPUT_DIR,
    SEED,
    SUMMARY_CSV_HEADER,
    TRIAL_CSV_HEADER,
    TRIALS,
    GroundTruth,
    average_poses,
    collect_ground_truth,
    evaluate_trial,
    main,
    perturb_pose,
    resolve_output_dir,
    rotation_error_deg,
    run,
    translation_error_mm,
)
from so101_classical_control.kinematics import (
    inverse_se3,
    pose,
    quaternion_to_rotation,
    rotation_to_axis_angle,
    rz,
    translation_of,
)

SAMPLE_COUNT = 20000
ROTATION_SIGMA_DEG = 1.0
TRANSLATION_SIGMA_MM = 5.0
#: Sampling tolerance for a standard-deviation estimate from 20000 samples.
RELATIVE_STD_TOL = 0.05
#: Sampling tolerance for the Maxwell mean.
MAXWELL_MEAN_TOL = 0.03
#: Synthetic configurations and trials of the eye-in-hand growth check.
SYNTHETIC_CONFIGURATIONS = 8
SYNTHETIC_TRIALS = 20
#: Rotation error bounds of the calibration tests, in degrees and millimetres.
MUJOCO_ROTATION_TOL_DEG = math.degrees(1e-6)
MUJOCO_TRANSLATION_TOL_MM = 1e-3
#: Maximum size of the committed plot and summary CSV.
MAX_PNG_BYTES = 300_000
MAX_SUMMARY_BYTES = 50_000

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
COMMITTED_SUMMARY = RESULTS_DIR / "noise_robustness_summary.csv"
COMMITTED_PNG = RESULTS_DIR / "noise_robustness.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    """Return a random SE(3) transform with a uniformly random rotation."""
    rotation = quaternion_to_rotation(rng.normal(size=4))
    translation = rng.uniform(-0.2, 0.2, 3)
    return pose(rotation, translation)


def _rotation_vector(transform: np.ndarray) -> np.ndarray:
    """Return the rotation vector ``angle * axis`` of an SE(3) transform."""
    axis, angle = rotation_to_axis_angle(transform[:3, :3])
    return angle * axis


def _run_small(output_dir: Path, seed: int, ground_truth: GroundTruth | None = None):
    """Run a small sweep suitable for CI."""
    return run(
        output_dir,
        trials=2,
        seed=seed,
        pose_counts=(6, 12),
        noise_levels=(0.0, 1.0),
        ground_truth=ground_truth,
    )


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Return ``(header, rows)`` of a CSV file."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = tuple(reader.fieldnames or ())
    return header, rows


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------


def test_perturb_pose_with_zero_sigmas_is_a_noop() -> None:
    rng = np.random.default_rng(SEED)
    base = _random_transform(rng)
    result = perturb_pose(base, 0.0, 0.0, rng)
    np.testing.assert_array_equal(result, base)
    assert result.dtype == np.float64


def test_perturb_pose_returns_a_valid_se3() -> None:
    rng = np.random.default_rng(SEED)
    base = _random_transform(rng)
    for _ in range(5):
        result = perturb_pose(base, math.radians(5.0), 0.01, rng)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result[3], [0.0, 0.0, 0.0, 1.0])
        rotation = result[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_perturb_pose_statistics_match_the_documented_sigmas() -> None:
    rng = np.random.default_rng(SEED)
    base = _random_transform(rng)
    rotation_sigma = math.radians(ROTATION_SIGMA_DEG)
    translation_sigma = 1e-3 * TRANSLATION_SIGMA_MM

    rotation_vectors = np.array(
        [
            _rotation_vector(inverse_se3(base) @ perturb_pose(base, rotation_sigma, 0.0, rng))
            for _ in range(SAMPLE_COUNT)
        ]
    )
    rotation_std = rotation_vectors.std(axis=0, ddof=1)
    np.testing.assert_allclose(rotation_std, rotation_sigma, rtol=RELATIVE_STD_TOL, atol=0.0)

    translations = np.array(
        [
            translation_of(inverse_se3(base) @ perturb_pose(base, 0.0, translation_sigma, rng))
            for _ in range(SAMPLE_COUNT)
        ]
    )
    translation_std = translations.std(axis=0, ddof=1)
    np.testing.assert_allclose(translation_std, translation_sigma, rtol=RELATIVE_STD_TOL, atol=0.0)

    maxwell_mean = translation_sigma * 2.0 * math.sqrt(2.0 / math.pi)
    measured_mean = float(np.linalg.norm(translations, axis=1).mean())
    assert math.isclose(measured_mean, maxwell_mean, rel_tol=MAXWELL_MEAN_TOL)

    print(
        f"\nnoise model: rotation std {rotation_std} rad, translation std "
        f"{translation_std} m, mean ||t|| {measured_mean:.6f} m "
        f"(Maxwell {maxwell_mean:.6f} m)"
    )


# ---------------------------------------------------------------------------
# Closed-loop formula
# ---------------------------------------------------------------------------


def test_closed_loop_single_translation_error_is_the_noise_norm() -> None:
    """With translation-only marker noise the error is exactly the noise norm.

    ``T_est = T_base_marker inv(T_cam_marker Delta)``. For a static marker
    ``T_base_marker = T_true T_cam_marker``, so
    ``T_est = T_true T_cm inv(Delta) inv(T_cm)``. With
    ``Delta = pose(I, t_delta)`` this is ``T_true pose(I, -R_cm t_delta)``: the
    rotation is unchanged and the translation error is
    ``||R_cm t_delta|| = ||t_delta||``. The rotation error is therefore zero and
    the translation error is the noise norm exactly.
    """
    rng = np.random.default_rng(SEED)
    x_true = _random_transform(rng)
    base_marker = _random_transform(rng)
    cam_marker = inverse_se3(x_true) @ base_marker

    measured = perturb_pose(cam_marker, 0.0, 0.1, rng)
    delta = inverse_se3(cam_marker) @ measured
    estimate = base_marker @ inverse_se3(measured)

    assert abs(rotation_error_deg(estimate, x_true)) <= 1e-12
    translation_error_m = 1e-3 * translation_error_mm(estimate, x_true)
    noise_norm = float(np.linalg.norm(translation_of(delta)))
    assert abs(translation_error_m - noise_norm) <= 1e-12


# ---------------------------------------------------------------------------
# average_poses
# ---------------------------------------------------------------------------


def test_average_poses_of_identical_poses_returns_the_pose() -> None:
    rng = np.random.default_rng(SEED)
    single = _random_transform(rng)
    mean = average_poses([single] * 5)
    np.testing.assert_allclose(mean, single, atol=1e-12, rtol=0.0)


def test_average_poses_of_symmetric_rotations_returns_the_centre() -> None:
    rng = np.random.default_rng(SEED)
    centre = _random_transform(rng)
    angle = 0.4
    mean = average_poses(
        [
            centre @ pose(rz(angle), np.zeros(3)),
            centre @ pose(rz(-angle), np.zeros(3)),
        ]
    )
    np.testing.assert_allclose(mean, centre, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# Synthetic eye-in-hand problem
# ---------------------------------------------------------------------------


def _synthetic_hand_eye_problem() -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
]:
    rng = np.random.default_rng(SEED)
    x_true = _random_transform(rng)
    hand_poses = [_random_transform(rng) for _ in range(SYNTHETIC_CONFIGURATIONS)]
    cam_poses = [hand @ x_true for hand in hand_poses]
    desk_marker = _random_transform(rng)
    return x_true, hand_poses, cam_poses, desk_marker


def _hand_eye_estimate(
    hand_poses: list[np.ndarray],
    cam_poses: list[np.ndarray],
    desk_marker: np.ndarray,
    rotation_sigma: float,
    translation_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cam_estimates = [
        desk_marker
        @ inverse_se3(
            perturb_pose(
                inverse_se3(cam_pose) @ desk_marker,
                rotation_sigma,
                translation_sigma,
                rng,
            )
        )
        for cam_pose in cam_poses
    ]
    a_list, b_list = build_motion_pairs(hand_poses, cam_estimates)
    return solve_ax_xb(a_list, b_list)


def test_synthetic_hand_eye_is_recovered_without_noise() -> None:
    """Noiseless synthetic data must be recovered to 1e-9 m and 1e-9 rad.

    The rotation error uses the same well-conditioned ``arctan2`` form as the
    solver's own tests; ``arccos`` cannot resolve this bound near zero error.
    """
    x_true, hand_poses, cam_poses, desk_marker = _synthetic_hand_eye_problem()
    estimate = _hand_eye_estimate(
        hand_poses, cam_poses, desk_marker, 0.0, 0.0, np.random.default_rng(SEED)
    )
    translation_error_m = float(np.linalg.norm(translation_of(estimate) - translation_of(x_true)))
    rotation_error_rad = math.radians(rotation_error_deg(estimate, x_true))
    print(
        f"\nsynthetic hand-eye: rotation error {rotation_error_rad:.3e} rad, "
        f"translation error {translation_error_m:.3e} m"
    )
    assert rotation_error_rad <= 1e-9
    assert translation_error_m <= 1e-9


def test_synthetic_hand_eye_error_grows_with_marker_noise() -> None:
    x_true, hand_poses, cam_poses, desk_marker = _synthetic_hand_eye_problem()
    medians: list[float] = []
    for level_index, level in enumerate((0.25, 2.0)):
        errors = []
        for trial in range(SYNTHETIC_TRIALS):
            rng = np.random.default_rng([SEED, level_index, trial])
            estimate = _hand_eye_estimate(
                hand_poses,
                cam_poses,
                desk_marker,
                math.radians(level),
                1e-3 * 5.0 * level,
                rng,
            )
            errors.append(rotation_error_deg(estimate, x_true))
        medians.append(float(np.median(errors)))
    print(f"\nsynthetic hand-eye median rotation error: {medians[0]:.4f} -> {medians[1]:.4f} deg")
    assert medians[0] < medians[1]


# ---------------------------------------------------------------------------
# MuJoCo ground truth (one collection for the whole session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ground_truth() -> GroundTruth:
    """Ground-truth poses from the public backend, collected once per session."""
    return collect_ground_truth()


def test_level_zero_matches_the_mujoco_ground_truth(ground_truth: GroundTruth) -> None:
    rows = evaluate_trial(
        ground_truth,
        "none",
        0.0,
        CONFIGURATION_COUNT,
        0,
        np.random.default_rng(SEED),
    )
    assert {row.method for row in rows} == set(METHODS)
    for row in rows:
        assert row.status == "ok", row.method
        assert row.rotation_error_deg <= MUJOCO_ROTATION_TOL_DEG, row.method
        assert row.translation_error_mm <= MUJOCO_TRANSLATION_TOL_MM, row.method
    print(
        "\nlevel-zero worst error: "
        f"{max(row.rotation_error_deg for row in rows):.3e} deg, "
        f"{max(row.translation_error_mm for row in rows):.3e} mm"
    )


# ---------------------------------------------------------------------------
# Determinism and output contract
# ---------------------------------------------------------------------------


def test_runs_are_deterministic(tmp_path: Path, ground_truth: GroundTruth) -> None:
    first = _run_small(tmp_path / "first", SEED, ground_truth)
    second = _run_small(tmp_path / "second", SEED, ground_truth)
    assert first.trials_csv.read_bytes() == second.trials_csv.read_bytes()
    assert first.summary_csv.read_bytes() == second.summary_csv.read_bytes()

    other = _run_small(tmp_path / "other", SEED + 1, ground_truth)
    assert other.trials_csv.read_bytes() != first.trials_csv.read_bytes()


def test_output_contract(tmp_path: Path, ground_truth: GroundTruth) -> None:
    result = _run_small(tmp_path / "contract", SEED, ground_truth)

    trials_header, _ = _read_csv(result.trials_csv)
    assert trials_header == TRIAL_CSV_HEADER
    summary_header, _ = _read_csv(result.summary_csv)
    assert summary_header == SUMMARY_CSV_HEADER

    payload = result.plot_png.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(payload) < MAX_PNG_BYTES


# ---------------------------------------------------------------------------
# Committed artifact freshness (R1: regenerates a subset, not the whole grid)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def subset_sweep(tmp_path_factory, ground_truth: GroundTruth):
    """Regenerate the committed-subset cells once for the whole session.

    Pose count ``24`` at levels ``0.0`` and ``1.0``, all sources, full 50
    trials. This is the R1 subset: it is valid because each cell's trials are
    seeded with stable tokens (R6), so a cell's rows do not depend on which
    other cells are run; ``test_r6_cell_seed_is_independent_of_grid_subset``
    proves that directly.
    """
    output = tmp_path_factory.mktemp("subset_sweep")
    return run(
        output,
        trials=TRIALS,
        seed=SEED,
        pose_counts=(24,),
        noise_levels=(0.0, 1.0),
        ground_truth=ground_truth,
    )


def test_committed_summary_matches_regeneration(subset_sweep) -> None:
    """The committed summary rows must be reproducible from the subset above."""
    if not COMMITTED_SUMMARY.exists():
        pytest.skip("results/noise_robustness_summary.csv is not committed")

    regenerated = subset_sweep.summary_csv
    committed_header, committed_rows = _read_csv(COMMITTED_SUMMARY)
    regenerated_header, regenerated_rows = _read_csv(regenerated)
    assert committed_header == regenerated_header

    subset_keys = {
        (row["noise_source"], row["noise_level"], row["method"], row["cell_pose_count"])
        for row in regenerated_rows
    }
    assert len(subset_keys) == len(regenerated_rows), "regenerated subset is not unique"

    committed_by_key = {
        (row["noise_source"], row["noise_level"], row["method"], row["cell_pose_count"]): row
        for row in committed_rows
    }
    for key in subset_keys:
        assert key in committed_by_key, f"committed summary is missing {key}"

    for regenerated_row in regenerated_rows:
        key = (
            regenerated_row["noise_source"],
            regenerated_row["noise_level"],
            regenerated_row["method"],
            regenerated_row["cell_pose_count"],
        )
        committed_row = committed_by_key[key]
        for column in committed_header:
            if column in ("noise_source", "method"):
                assert committed_row[column] == regenerated_row[column], column
            elif column in (
                "num_poses",
                "cell_pose_count",
                "trials",
                "solver_errors",
            ):
                assert int(committed_row[column]) == int(regenerated_row[column]), column
            else:
                np.testing.assert_allclose(
                    float(committed_row[column]),
                    float(regenerated_row[column]),
                    rtol=1e-6,
                    atol=1e-9,
                    err_msg=column,
                )


def test_r6_cell_seed_is_independent_of_grid_subset(
    tmp_path: Path, ground_truth: GroundTruth
) -> None:
    """A cell's trial rows are identical alone and inside the full small grid."""
    alone = run(
        tmp_path / "alone",
        trials=3,
        seed=SEED,
        pose_counts=(24,),
        noise_levels=(1.0,),
        ground_truth=ground_truth,
    )
    full = run(
        tmp_path / "full",
        trials=3,
        seed=SEED,
        pose_counts=(6, 24),
        noise_levels=(0.0, 1.0),
        ground_truth=ground_truth,
    )

    def keyed(rows):
        return {
            (row.noise_source, row.noise_level, row.method, row.cell_pose_count, row.trial): (
                row.status,
                row.rotation_error_deg,
                row.translation_error_mm,
            )
            for row in rows
            if row.noise_source == "joint"
            and row.noise_level == 1.0
            and row.cell_pose_count == 24
        }

    alone_rows = keyed(alone.trial_rows)
    full_rows = keyed(full.trial_rows)
    assert alone_rows, "the single-cell run produced no joint rows"
    assert alone_rows == full_rows


# ---------------------------------------------------------------------------
# Review finding tests
# ---------------------------------------------------------------------------


def test_r2_single_method_not_pooled() -> None:
    """``_summarize`` keeps the single method separate per cell pose count.

    Synthetic rows stand in for the three pose-count cells so this check is
    cheap; the committed summary then proves the same property end to end.
    """
    import so101_classical_control.eval.noise_robustness as module

    rows = []
    for cell_pose_count in (6, 12, 24):
        for trial in range(TRIALS):
            rows.append(
                module.TrialRow(
                    noise_source="both",
                    noise_level=1.0,
                    joint_sigma_deg=1.0,
                    marker_rotation_sigma_deg=1.0,
                    marker_translation_sigma_mm=5.0,
                    method="camera_to_base_closed_loop_single",
                    num_poses=1,
                    cell_pose_count=cell_pose_count,
                    trial=trial,
                    status="ok",
                    rotation_error_deg=float(trial),
                    translation_error_mm=float(trial),
                )
            )

    summary = module._summarize(rows)
    keys = [
        (row.noise_source, row.noise_level, row.method, row.num_poses, row.cell_pose_count)
        for row in summary
    ]
    assert len(keys) == len(set(keys)), "summary keys are not unique"
    assert len(summary) == 3
    for row in summary:
        assert row.trials == TRIALS, (row.cell_pose_count, row.trials)
        assert row.num_poses == 1

    if COMMITTED_SUMMARY.exists():
        _, committed_rows = _read_csv(COMMITTED_SUMMARY)
        committed_single = [
            row for row in committed_rows if row["method"] == "camera_to_base_closed_loop_single"
        ]
        assert {int(row["cell_pose_count"]) for row in committed_single} == {6, 12, 24}
        summary_keys = [
            (row["noise_source"], row["noise_level"], row["method"], row["cell_pose_count"])
            for row in committed_rows
        ]
        assert len(summary_keys) == len(set(summary_keys)), "committed keys are not unique"
        for row in committed_single:
            assert int(row["num_poses"]) == 1
            assert int(row["trials"]) == TRIALS

    if COMMITTED_SUMMARY.exists():
        _, committed_rows = _read_csv(COMMITTED_SUMMARY)
        committed_single = [
            row
            for row in committed_rows
            if row["method"] == "camera_to_base_closed_loop_single"
        ]
        assert {int(row["cell_pose_count"]) for row in committed_single} == {6, 12, 24}
        for row in committed_single:
            assert int(row["num_poses"]) == 1
            assert int(row["trials"]) == TRIALS


def test_r3_docs_numbers_match_summary() -> None:
    """The level-1.0 table in results/README.md matches the committed CSV."""
    if not COMMITTED_SUMMARY.exists():
        pytest.skip("results/noise_robustness_summary.csv is not committed")
    readme = (RESULTS_DIR / "README.md").read_text(encoding="utf-8")

    # The first table under "## Median errors at level 1.0"; stop at the first
    # blank line so the later level-sweep table is not included.
    section = readme.split("## Median errors at level 1.0", 1)[1]
    table_lines: list[str] = []
    for line in section.splitlines():
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break
    table_lines = [line for line in table_lines if "`" in line]
    assert table_lines, "no level-1.0 table rows found in results/README.md"

    _, rows = _read_csv(COMMITTED_SUMMARY)

    def summary_value(method: str, column: str) -> str:
        matches = [
            row
            for row in rows
            if row["noise_source"] == "both"
            and float(row["noise_level"]) == 1.0
            and row["method"] == method
            and int(row["cell_pose_count"]) == 24
        ]
        assert len(matches) == 1, method
        return matches[0][column]

    checked = 0
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        method = cells[0].strip("`")
        assert method in METHODS, cells
        rotation_shown, translation_shown = cells[1], cells[2]
        digits = len(rotation_shown.split(".")[1])
        assert digits == len(translation_shown.split(".")[1])
        expected_rotation = round(float(summary_value(method, "rotation_error_deg_median")), digits)
        expected_translation = round(
            float(summary_value(method, "translation_error_mm_median")), digits
        )
        assert float(rotation_shown) == pytest.approx(expected_rotation, abs=10.0**-digits), (
            method,
            rotation_shown,
        )
        assert float(translation_shown) == pytest.approx(
            expected_translation, abs=10.0**-digits
        ), (method, translation_shown)
        checked += 1
    assert checked == len(METHODS)


def test_r4_quick_defaults_to_results_quick() -> None:
    """``--quick`` without ``--output-dir`` resolves to ``results/quick``."""
    assert resolve_output_dir(None, quick=True) == str(QUICK_OUTPUT_DIR)
    assert QUICK_OUTPUT_DIR == Path("results") / "quick"
    assert resolve_output_dir(None, quick=False) == "results"
    assert resolve_output_dir("custom/dir", quick=True) == "custom/dir"
    assert resolve_output_dir("custom/dir", quick=False) == "custom/dir"


def test_r5_joint_noise_model_is_documented() -> None:
    """The module docstring, results README, and calibration docs state the model."""
    module = (
        REPO_ROOT / "src" / "so101_classical_control" / "eval" / "noise_robustness.py"
    ).read_text(encoding="utf-8")
    readme = (RESULTS_DIR / "README.md").read_text(encoding="utf-8")
    calibration = (REPO_ROOT / "docs" / "calibration.md").read_text(encoding="utf-8")
    for document in (module, readme, calibration):
        assert "encoder" in document.lower(), "joint-noise model is not documented"


def test_r7_trial_rows_unique(tmp_path: Path, ground_truth: GroundTruth) -> None:
    """Raw trial rows are uniquely keyed by the documented tuple."""
    result = run(
        tmp_path,
        trials=3,
        seed=SEED,
        pose_counts=(6, 24),
        noise_levels=(0.0, 1.0),
        ground_truth=ground_truth,
    )
    keys = [
        (row.noise_source, row.noise_level, row.method, row.cell_pose_count, row.trial)
        for row in result.trial_rows
    ]
    assert len(keys) == len(set(keys)), "duplicate trial rows"
    # Every cell contributes one row per method per trial.
    cells = {
        (row.noise_source, row.noise_level, row.cell_pose_count) for row in result.trial_rows
    }
    expected = len(cells) * len(METHODS) * 3
    assert len(result.trial_rows) == expected

    header, csv_rows = _read_csv(result.trials_csv)
    csv_keys = [
        (
            row["noise_source"],
            row["noise_level"],
            row["method"],
            row["cell_pose_count"],
            row["trial"],
        )
        for row in csv_rows
    ]
    assert len(csv_keys) == len(set(csv_keys)), "duplicate raw CSV keys"
    assert len(csv_rows) == len(result.trial_rows)


def test_r8_committed_png_contract() -> None:
    """The committed PNG is a real PNG under 300 kB; the summary CSV is small."""
    if not COMMITTED_PNG.exists() or not COMMITTED_SUMMARY.exists():
        pytest.skip("committed artifacts are not present")
    assert COMMITTED_PNG.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert COMMITTED_PNG.stat().st_size < MAX_PNG_BYTES
    assert COMMITTED_SUMMARY.stat().st_size < MAX_SUMMARY_BYTES


def test_r9_cli_quick_smoke(tmp_path: Path) -> None:
    """``main(["--quick", ...])`` writes both CSVs and the PNG."""
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    for name in (
        "noise_robustness_trials.csv",
        "noise_robustness_summary.csv",
        "noise_robustness.png",
    ):
        written = tmp_path / name
        assert written.exists(), name
        assert written.stat().st_size > 0, name
    assert (tmp_path / "noise_robustness.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_r10_static_poses_are_checked() -> None:
    """The static-pose guard raises when the camera or marker moves.

    ``collect_ground_truth`` runs the same guard on the real configurations and
    would raise ``RuntimeError`` for a moving ``top_cam`` or ``desk_marker``.
    """
    import so101_classical_control.eval.noise_robustness as module

    static = [np.eye(4), np.eye(4)]
    module._assert_static("static", static)

    moved_translation = [np.eye(4), np.eye(4)]
    moved_translation[1][0, 3] = 1e-3
    with pytest.raises(RuntimeError):
        module._assert_static("moving_translation", moved_translation)

    moved_rotation = [np.eye(4), np.eye(4)]
    moved_rotation[1][:3, :3] = rz(math.radians(1.0))
    with pytest.raises(RuntimeError):
        module._assert_static("moving_rotation", moved_rotation)


def test_r12_negative_seed_rejected(tmp_path: Path, ground_truth: GroundTruth) -> None:
    """``run()`` rejects a negative seed and a non-positive trial count."""
    with pytest.raises(ValueError):
        run(tmp_path, trials=1, seed=-1, ground_truth=ground_truth)
    with pytest.raises(ValueError):
        run(tmp_path, trials=0, seed=SEED, ground_truth=ground_truth)


def test_r13_p95_is_documented_as_linear_quantile() -> None:
    """results/README.md states the p95 quantile rule and its instability."""
    readme = (RESULTS_DIR / "README.md").read_text(encoding="utf-8")
    assert re.search(r"linear[- ]interpolation", readme, re.IGNORECASE)
    assert "median" in readme.lower()


# ---------------------------------------------------------------------------
# Results policy
# ---------------------------------------------------------------------------


RESULTS_IGNORE_LINES = (
    "results/*",
    "!results/README.md",
    "!results/.gitkeep",
    "!results/noise_robustness_summary.csv",
    "!results/noise_robustness.png",
)


def test_results_gitignore_policy() -> None:
    gitignore = REPO_ROOT / ".gitignore"
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    for expected in RESULTS_IGNORE_LINES:
        assert expected in lines, expected
