"""Evaluation metrics and experiment runners.

``noise_robustness.py`` is the Issue #10 noise-robustness sweep: it perturbs the
ground-truth joint angles and marker poses with synthetic Gaussian noise and
reports the error of both hand-eye calibrations (eye-in-hand ``A X = X B`` and
camera-to-base) over a noise grid. It reuses the calibration solvers unchanged
and writes the CSVs and the plot under ``results/``.

Keep this module import light: the sweep module is not imported here, and
``matplotlib`` is imported only inside its plotting function. Run the sweep with
``uv run python -m so101_classical_control.eval.noise_robustness``.
"""
