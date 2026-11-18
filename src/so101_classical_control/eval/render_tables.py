"""Deterministic Markdown tables rendered from the committed summary CSV.

This module turns ``results/noise_robustness_summary.csv`` into the Markdown
block that the documentation files embed between two marker lines. The tables
are a pure function of the CSV, so ``--check-docs`` can regenerate them in CI
and fail when a committed document drifts from its data.

The generated block has two tables:

- the level ``1.0`` comparison for source ``both`` and cell pose count ``24``,
  one row per method, with the median and p95 rotation and translation errors;
- the level sweep for the same source and cell, levels ``0.25``, ``1.0``, and
  ``2.0``, with the medians shown as ``rotation / translation``.

Every number is formatted with :func:`_format_sig`, ``f"{value:.3g}"``, i.e.
three significant digits. The caption is a single line stating the condition
the tables describe.

Only ``csv`` and ``pathlib`` are used, so the renderer does not import MuJoCo or
the sweep module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

__all__ = [
    "CONDITION_CAPTION",
    "MARKER_BEGIN",
    "MARKER_END",
    "METHOD_ORDER",
    "render_summary_tables",
    "update_generated_block",
]

#: Exact marker lines that delimit the generated block in a document.
MARKER_BEGIN: Final[str] = "<!-- BEGIN generated: noise_robustness tables -->"
MARKER_END: Final[str] = "<!-- END generated: noise_robustness tables -->"

#: Reporting order of the four methods in both tables.
METHOD_ORDER: Final[tuple[str, ...]] = (
    "hand_eye_axxb",
    "camera_to_base_closed_loop_single",
    "camera_to_base_closed_loop_mean",
    "camera_to_base_axxb",
)

#: The noise condition the generated tables describe.
CONDITION_CAPTION: Final[str] = (
    "Source `both`, cell pose count N = 24, 50 trials per cell; "
    "`camera_to_base_closed_loop_single` is 1 pose per trial drawn from the "
    "cell's 24 configurations."
)

_SOURCE: Final[str] = "both"
_CELL_POSE_COUNT: Final[int] = 24
_LEVEL_FOR_METHOD_TABLE: Final[float] = 1.0
_SWEEP_LEVELS: Final[tuple[float, ...]] = (0.25, 1.0, 2.0)


def _format_sig(value: float) -> str:
    """Format ``value`` with three significant digits: ``f"{value:.3g}"``."""
    return f"{value:.3g}"


def _read_rows(summary_csv: Path) -> list[dict[str, str]]:
    """Return the summary rows as dictionaries keyed by the CSV header."""
    with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select(
    rows: list[dict[str, str]], method: str, level: float
) -> dict[str, str]:
    """Return the single row for one method and level in the documented cell."""
    matches = [
        row
        for row in rows
        if row["noise_source"] == _SOURCE
        and float(row["noise_level"]) == level
        and int(row["cell_pose_count"]) == _CELL_POSE_COUNT
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one summary row for method {method!r} at level "
            f"{level:g} in cell N = {_CELL_POSE_COUNT}, found {len(matches)}"
        )
    return matches[0]


def render_summary_tables(summary_csv: Path) -> str:
    """Return the deterministic Markdown block for a summary CSV.

    The result contains :data:`CONDITION_CAPTION`, the level ``1.0`` method
    table, the ``0.25``/``1.0``/``2.0`` sweep table, and no trailing newline.
    Every numeric cell is formatted by :func:`_format_sig`, three significant
    digits.

    Raises
    ------
    ValueError
        If a method/level row is missing or duplicated in the CSV.
    """
    rows = _read_rows(summary_csv)

    lines: list[str] = [CONDITION_CAPTION, ""]
    lines.append(
        "| Method | num_poses | median rotation deg | median translation mm | "
        "p95 rotation deg | p95 translation mm |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for method in METHOD_ORDER:
        row = _select(rows, method, _LEVEL_FOR_METHOD_TABLE)
        rotation_median = _format_sig(float(row["rotation_error_deg_median"]))
        translation_median = _format_sig(float(row["translation_error_mm_median"]))
        rotation_p95 = _format_sig(float(row["rotation_error_deg_p95"]))
        translation_p95 = _format_sig(float(row["translation_error_mm_p95"]))
        lines.append(
            f"| `{method}` | {int(row['num_poses'])} | {rotation_median} | "
            f"{translation_median} | {rotation_p95} | {translation_p95} |"
        )

    lines.append("")
    header_cells = [
        f"rot / trans (s={level:g})" for level in _SWEEP_LEVELS
    ]
    lines.append("| Method | " + " | ".join(header_cells) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in _SWEEP_LEVELS) + " |")
    for method in METHOD_ORDER:
        cells = []
        for level in _SWEEP_LEVELS:
            row = _select(rows, method, level)
            rotation = _format_sig(float(row["rotation_error_deg_median"]))
            translation = _format_sig(float(row["translation_error_mm_median"]))
            cells.append(f"{rotation} / {translation}")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def update_generated_block(doc_text: str, block: str) -> str:
    """Replace the content between the generated-block markers.

    The marker lines themselves are kept. ``block`` is inserted with exactly one
    newline after :data:`MARKER_BEGIN` and one before :data:`MARKER_END`, so
    applying this function to its own output is idempotent.

    Raises
    ------
    ValueError
        If either marker is missing or appears more than once, or if
        :data:`MARKER_END` precedes :data:`MARKER_BEGIN`.
    """
    begin_count = doc_text.count(MARKER_BEGIN)
    end_count = doc_text.count(MARKER_END)
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "generated-block markers must appear exactly once: "
            f"{MARKER_BEGIN!r} found {begin_count} times, "
            f"{MARKER_END!r} found {end_count} times"
        )

    begin_index = doc_text.index(MARKER_BEGIN)
    end_index = doc_text.index(MARKER_END)
    if end_index < begin_index:
        raise ValueError(f"{MARKER_END!r} appears before {MARKER_BEGIN!r}")

    prefix = doc_text[: begin_index + len(MARKER_BEGIN)]
    suffix = doc_text[end_index:]
    return prefix + "\n" + block.rstrip("\n") + "\n" + suffix
