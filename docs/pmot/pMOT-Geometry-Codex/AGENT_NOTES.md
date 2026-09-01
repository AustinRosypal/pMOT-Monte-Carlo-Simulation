# Agent Notes

## Chosen Vertical Path Solution

The chosen vertical-path solution is `FinalVerticalSolution.ipynb`.

It uses the left-injection vertical geometry with:

- `L1 = AC254-045-C`
- `L2 = AC254-045-C`
- normal-incidence cell orientation for this vertical path
- left optical-table reference shifted outward by `25.4 mm`
- no optics inside the vertical coil exclusion region `|z| < 36 mm`
- final plot: `pmot_combined_mixed_lenses_geometry_l1_045_l2_045.png`

This is the working vertical solution. Earlier vertical or normal-incidence exploration files are historical and should not be used to infer the final vertical layout unless the user explicitly asks to revisit them.

The older compact vertical variants using `AC254-035-C`, `AC254-040-C`, mixed lens pairs, cube-fit tests, or right-angle mirror variants are irrelevant for the currently chosen vertical path.

## Horizontal Paths

Horizontal paths are a separate problem and should not inherit the vertical coil-exclusion constraints.

For the horizontal paths:

- the cell is at `45 deg` AOI
- use the `AC508-080-C` achromat, `f = 80.3 mm`
- the only hard geometric exclusion currently modeled is the glass cell itself
- the existing 1530 round-trip target foci remain `-10 mm` and `+10 mm`
- the 780 overlay is the horizontal cat-eye style overlay with a `12.7 mm` MOT beam unless the user changes that assumption

Use `FinalHorizontalSolution.ipynb` / `pmot_final_horizontal_solution.py` for the current horizontal-path solution.
