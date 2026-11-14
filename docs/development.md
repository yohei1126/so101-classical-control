# Development

## Environment Requirements

Python requirement: `>=3.11` (CI runs on Python 3.11).

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

Tests must run headless.

## Current Tests

The test suite currently contains import smoke tests in `tests/test_import.py` for the eight subpackages.
