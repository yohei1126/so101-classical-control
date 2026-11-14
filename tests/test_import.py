"""Smoke tests that every package in the scaffold can be imported."""

import importlib

import pytest

SUBPACKAGES = [
    "kinematics",
    "calibration",
    "perception",
    "planning",
    "control",
    "sim",
    "real",
    "eval",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import_subpackage(name: str) -> None:
    """Each subpackage imports and carries a docstring."""
    module = importlib.import_module(f"so101_classical_control.{name}")
    assert module.__doc__ is not None
