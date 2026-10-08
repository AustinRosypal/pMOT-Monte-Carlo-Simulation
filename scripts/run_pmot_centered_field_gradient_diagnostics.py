"""Generate centered equal-power surrogate-field and gradient diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pmot.pmot.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    surrogate_field_jacobians_t_per_m,
)
from pmot.pmot.surrogate_effective_field_study import create_field_diagnostics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = (
        project_root
        / "outputs"
        / "diagnostics"
        / "pmot"
        / "MOT testing with a strange defined magnetic field"
        / "centered equal-power field 20261002"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = create_field_diagnostics(
        output,
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
        plane_samples=241,
    )
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    cfg = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    legacy = LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG
    axis_gradient_rows = []
    for radius_mm in (0.5, 1.0, 2.0, 3.0):
        points = np.zeros((3, 3))
        points[np.arange(3), np.arange(3)] = radius_mm * 1.0e-3
        jacobians = 100.0 * surrogate_field_jacobians_t_per_m(
            points,
            step_m=1.0e-6,
            config=cfg,
        )
        values = [float(jacobians[index, index, index]) for index in range(3)]
        axis_gradient_rows.append((radius_mm, values))
    gradient_lines = "\n".join(
        f"- At {radius_mm:.1f} mm: {values} G/cm."
        for radius_mm, values in axis_gradient_rows
    )
    report = f"""# Centered equal-power surrogate effective field

The forward and return traveling powers remain equal. The solved waist
locations are retained at {1e3*cfg.forward_waist_position_m:.8f} mm and
{1e3*cfg.return_waist_position_m:.8f} mm.

## Rayleigh-range choice

- Forward: {1e6*cfg.forward_rayleigh_range_m:.9f} micrometers.
- Return: {1e6*cfg.return_rayleigh_range_m:.9f} micrometers.
- Corresponding waist radii: approximately 2.105546 and 2.093651 micrometers.

The pair satisfies equal on-axis intensity at the origin exactly while
preserving the geometric mean of the former Rayleigh ranges
({1e6*(legacy.forward_rayleigh_range_m*legacy.return_rayleigh_range_m)**0.5:.9f}
micrometers). This is the unique choice that makes the logarithmic fractional
changes equal and opposite under those two constraints.

## Central field

- Effective field at the origin [G]: {summary['origin_field_g']}.
- Diagonal gradient [G/cm]:
  {[summary['origin_jacobian_g_per_cm'][i][i] for i in range(3)]}.
- Divergence proxy at the origin [G/cm]:
  {summary['origin_divergence_g_per_cm']}.

The component cuts use a smooth asinh display scale. The explicit gradient
plots cover the central +/-3 mm in the XY, XZ, and YZ planes. All numerical
arrays are saved under `data/effective_field`.

## Interpretation of the form

The zero and the desired local diagonal form are achieved at the origin. Along
the three principal axes the gradients evolve as follows:

{gradient_lines}

The plane maps also show that the field is not a spatially uniform ideal
quadrupole throughout the full +/-3 mm region. The finite transverse Gaussian
envelopes generate off-diagonal derivatives and curvature away from the axes.
Thus the chosen pair fixes the central bias exactly and gives the requested
local gradient, but it does not remove the intrinsic three-dimensional
Gaussian-envelope structure.

This remains a stretched-transition-equivalent surrogate field, not a
state-resolved AC-Stark Hamiltonian.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
