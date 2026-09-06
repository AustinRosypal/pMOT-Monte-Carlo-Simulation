# Final QA report: refined multilevel MOT relationship campaign

Date: 2026-09-05  
Outcome: **PASS**

## Scope

This campaign uses the authoritative 24-state, repumper-included,
adiabatic population-rate-equation Rb-87 MOT. The three loading sweeps are
independent parameter scans rather than a Cartesian product. Every loading
point uses 25 randomly sampled full-sphere incident-direction discs and 25
uniform-area launch points per 15 mm-radius disc. Direction discs are the 25
independent clusters used for pointwise 95% Student-t intervals (24 degrees of
freedom). The same seeded geometry is shared across scan points so only the
requested parameter changes; this common-random-number design correlates
neighboring points.

The temperature scan uses 25 independent preloaded Langevin clouds of 25 atoms
at every detuning. The deterministic force scan uses one rate-equation
calculation at each of 111 detunings, with coarse/fine numerical checks and
gravity excluded.

## Completed grids and selected results

| Study | Grid | Number of points | Selected result |
|---|---:|---:|---:|
| Loading vs. on-resonance saturation | requested values from 0.25 through 125 | 24 | at `s0=125`: 69.9947 million atoms/s, 95% CI [62.2568, 77.7327] million atoms/s |
| Loading vs. effective saturation | 0.25 through 5.00 by 0.25 | 20 | at `s_eff=5`: 69.9654 million atoms/s, 95% CI [62.2563, 77.6745] million atoms/s |
| Loading vs. detuning | `Delta/Gamma=-0.5` through `-6.0` by `-0.25` | 23 | maximum sampled mean at `-3.5`: 81.7985 million atoms/s, 95% CI [70.2365, 93.3604] million atoms/s |
| Finite-time temperature estimate | same 23 detunings | 23 | minimum sampled estimate at `-1.25`: 596.261 microkelvin, 95% CI [588.518, 604.005] microkelvin |
| Restoring slope and damping turnaround | `-0.5` through `-6.0` by `-0.05` | 111 | all Cartesian coarse/fine convergence and interior-turnaround checks pass |

The analytical temperature overlay uses

`T_D = -hbar*Gamma^2/(8*k_B*Delta) * [1+s_eff+(2*Delta/Gamma)^2]`,

with the single-cooling-beam Gaussian-center convention and `s_eff` recomputed
at each detuning. Its minimum on the sampled grid is 304.766 microkelvin at
`Delta/Gamma=-1.5`.

## Timeout correction and immutable provenance

The initial 50 ms capture search left endpoint classifications that required
longer observation. Two transactional post-campaign audits were applied:

- Detuning: 225 samples across 11 points; revision
  `83717017fca6aaa5fcff6986cb09bcff2df782d3814c8451f1210f26e83fc7f7`.
- Raw saturation: 155 samples across 4 points; revision
  `3dfa82f82e0b3edf1a6cceb804e56460c8e117e07db07bfc4714e9465143c4d5`.

Both used documented 200 ms coarse/fine checks, preserved the original run
signatures, verified pre/post hashes, and retained complete production-file
backups (114 and 44 files, respectively). The finalized production rows contain
no timeout classifications.

At `s0=0.25`, two near-edge rays are genuinely nonmonotone rather than scalar
capture thresholds:

- disc 6, point 22 captures at 0.50, 0.75, 1.00, and 1.25 m/s;
- disc 17, point 23 captures at 0.75 and 1.00 m/s.

Their spectra and loading contributions use the exact dual-timestep boolean
masks stored in the immutable audit. Their scalar CSV rows are explicit
zero-capture compatibility fallbacks and are excluded from the corrected
integrals. The resulting aggregate captured counts at 0, 0.25, 0.50, 0.75,
1.00, 1.25, and 1.50 m/s are 623, 623, 624, 625, 625, 623, and 621.

## Independent numerical QA

- Recomputed all 67 loading points from all 41,875 final capture rows,
  including velocity-resolved masks.
- Every saved capture spectrum, per-disc loading integral, mean, sample
  standard deviation, cluster SEM, Student-t confidence interval, point JSON,
  and aggregate CSV value matches the independent reconstruction. Maximum
  absolute floating residual: `1.11e-16`.
- Every loading point contains exactly 625 rows. Final and partial sample CSVs
  are identical at every point.
- All studies use geometry SHA-256
  `02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17`.
  Exact seeded regeneration succeeds; x, y, and z direction components all span
  both signs. The uniform-area diagnostic gives mean `(s/R)^2 = 0.52105`, and
  the full-sphere direction diagnostic gives `KS p = 0.361`.
- Mean loading across the 25 per-disc integrals equals loading obtained by
  integrating the mean capture spectrum, as required by linearity.
- All 14,375 temperature trajectories completed without failure and remained
  in the final 2 mm core.
- All 111 force rows pass coarse/fine convergence and interior-turnaround
  checks on x, y, and z.
- Final focused multilevel automated suite: 145/145 passed in 16.74 s.
- Final full repository automated suite: 306/306 passed in 37.97 s, with no
  warnings.

## Temperature interpretation

All 23 aggregate temperature points are marked `nonstationary`: 313 of the 575
individual clouds pass the stationarity test, but no detuning passes the strict
all-25-cloud gate. The plotted values are therefore finite-25-ms final-window
temperature estimates, not validated equilibrium temperatures. The survivor
fraction of one means all initially preloaded atoms remained within the final
2 mm core; it is not a loading or incident-capture fraction. The analytical
Doppler curve is a reference, not a guaranteed lower bound for these finite,
multilevel, provisional-model estimates.

## Force interpretation

The strongest sampled restoring slope occurs at `Delta/Gamma=-1.15`:
`x=y=-4.56215e-19 N/m` and `z=-9.12410e-19 N/m`. The damping-turnaround speed
rises from 3.513 m/s (`x/y`) and 4.270 m/s (`z`) at `-0.5` to 28.412 m/s
(`x/y`) and 28.419 m/s (`z`) at `-6.0`. Force-plot whiskers are absolute
fine-minus-coarse numerical-resolution differences, not statistical error
bars.

The inherited production force observable is the
ground-population-weighted available-absorption convention. It is retained for
comparison with prior multilevel campaigns but is not yet a validated net
scattering force; optical coherences, sub-Doppler physics, and a solver-wide
event-engine comparison remain outside this campaign.

## Final figures

- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/01_raw_saturation/loading_rate_vs_saturation_parameter.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/02_effective_saturation/loading_rate_vs_effective_saturation_parameter.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/03_detuning/loading_rate_vs_detuning.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/03_detuning/temperature/temperature_vs_detuning.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/03_detuning/temperature/temperature_only_vs_detuning.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/04_force_vs_detuning_27mW/damping_turnaround_vs_detuning.png`
- `outputs/figures/mot_multilevel/refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903/04_force_vs_detuning_27mW/restoring_slope_vs_detuning.png`
