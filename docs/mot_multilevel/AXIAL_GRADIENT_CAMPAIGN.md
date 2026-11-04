# Axial magnetic-gradient loading sweep, 2026-10-16

The user requested the existing 10, 15, 20, 25 and 30 G/cm study with an
additional 5 G/cm point, changing cooling detuning to -21 MHz. All six
settings must be newly simulated. The October 4 diagnostic script remains
unchanged for provenance; no completed outputs from that script were found.

## Active campaign

- Runner: `scripts/run_multilevel_gradient_campaign.py`.
- Command: `uv run python -u scripts/run_multilevel_gradient_campaign.py all --workers 16`.
- Statistics: `outputs/statistics/mot_multilevel_population_rate_v1/axial_gradient_5_10_15_20_25_30Gpcm_minus21MHz_20261016`.
- Figures: same campaign name under `outputs/figures/mot_multilevel_population_rate_v1`.
- Main figure: `loading_rate_vs_axial_gradient.png` and PDF; also save the
  density-independent loading coefficient and its error bars.

Use the repository-local environment installed with `uv sync --all-extras`.
No legacy environment path or historical diameter-script bootstrap is used.
No changes to the corrected force or trajectory source were needed. The
runner records current source, runner and uv-lock hashes, package versions,
compiled inputs and a source snapshot. The current checkout differs from
previous campaigns, so old certificates are not silently reused.

## Fixed settings

- Rb-87 D2, 8 ground and 16 excited states; corrected Section-12 stimulated
  rates and net absorption-minus-stimulated-emission force.
- Six cooling traveling components, 27 mW each, -21 MHz cooling detuning.
- Six repump traveling components, 0.1 mW each, zero repump detuning.
- Both families have 12.7 mm Gaussian 1/e² diameter and propagation-frame
  sigma+ helicities on x/y, sigma- on z, for incident and retro components.
- Change only coil current to achieve axial gradient magnitude 5, 10, 15,
  20, 25 or 30 G/cm. The default anti-Helmholtz geometry is retained.
- Gravity enabled; recoil diffusion disabled; instantaneous local steady
  state at every force evaluation. Temperature remains deferred.
- Full-sphere incident directions; uniform-area offsets on 15 mm-radius
  discs centered 15 mm from the trap; parallel velocities on each disc.
- Preserve the historical vapor prefactor and thermal scale. Report R/n87
  separately to support other densities; do not silently renormalize.

## Sampling, validation and continuation

Independent 32-disc x 20-point pilots use seed 2026101601. Their between-disc
loading variance determines a common production count, at least 100 discs,
rounded upward in blocks of 25, with a 1.5 variance safety factor. Production
uses independent seed 2026101602 and 20 points/disc. The count is frozen
before production. All gradients use common geometry for comparisons.
The target is pointwise 95% Student-t direction-disc half-width <=10%.
Do not repeatedly add samples until a conventional interval happens to pass.
If precision fails, preserve that result and plan an independent fixed-size
confirmation with explicit provenance. Sample count alone proves nothing.

Use the established `campaign_execution` positive-speed masks: 0.25--80 m/s
in 0.25 m/s steps plus 85--120 m/s tail probes, two adaptive settings at each
node, fixed-RK4 checks around all capture edges and disagreements, and the
retained duration/timestep escalation ladder through 2 s. Preserve
nonmonotone masks and every attempted node. Never relabel timeouts as escape.
Zero speed is only the exact zero weighted-flux anchor, not an imputed cross
section. Every setting must pass the nested-quadrature <=2% check and
representative longer-retention checks at two timesteps.

Before production: the 61 selected core/trajectory/capture/statistical tests
passed; 96 local field/force/population checks and six frozen-environment
event comparisons passed at the actual gradients and detuning. Spawned
worker checks compare 48 adaptive/fixed nodes. Two predetermined pilot rays
per gradient receive full fixed-RK4 checks at all 328 nodes (3,936 total).
Both spawned and full-grid gates must pass before production.

The shared worker cache originally keys only cooling power and detuning.
The new gradient wrapper clears it per ray and uses immutable arrays prepared
by the parent for the actual gradient; ARC never runs in a worker. A kernel
cache is isolated by current source digest. A nonblocking OS file lock stops
duplicate runner pools even if two continuations race.

Check `process.json`, `launcher.json` and actual live commands before any
restart. Other Python processes on this machine serve laboratory/Jupyter
work and must not be interrupted. Watch `run.log`, `run.stderr.log`,
`progress.json`, `pilot_progress.json`, `sampling_plan.json`, `failure.json`
and `completion.json`. Resume only with an identical manifest. Any source or
runner change requires a recorded revision and appropriate revalidation.
Preserve old logs on restart. If the run is healthy, keep notifications quiet.

After pilots, report the measured production ETA. After all six settings,
inspect both PNGs, verify aggregate audits and precision/retention gates,
write a visual-review record with hashes, report the six values, uncertainties,
sample sizes and runtime, link plots and report, and pause the heartbeat.
Do not claim completion while an issue or visual review remains pending.

Numerical checks qualify calculations within the local-steady-state model.
The weak-wing population-relaxation limitation remains unresolved; frozen-point
event agreement is not validation of moving-atom population adiabaticity.
Pointwise sampling intervals exclude model, vapor and apparatus uncertainty.

## Completion, 2026-10-16

The six settings finished at 23:29:59 UTC (6:29:59 PM Central), 115.31 minutes
after launch, including pilots and validation. The frozen allocation was
300 discs x 20 points per setting (36,000 production ray/setting calculations).
All six settings passed the 10% pointwise precision target; observed relative
95% half-widths range from 3.39% to 6.55%. All 3,595 representative retention
cases passed. The largest nested quadrature sensitivity was 0.0544%.

Loading means in gradient order 5, 10, 15, 20, 25, 30 G/cm are respectively
47.3531, 88.3505, 106.2290, 105.6820, 89.6575, 69.9946 million atoms/s.
Their pointwise 95% half-widths are 2.3866, 2.9939, 3.9928, 4.8724, 4.8465,
4.5835 million atoms/s. The paired 15-minus-20 G/cm difference is
0.5470 +/- 1.7124 million atoms/s (95%); it does not resolve a preference
between these two sampled settings. No continuous optimum is claimed.

`scripts/finalize_multilevel_gradient_campaign.py` checks source/runner/
snapshot/lockfile hashes, saved aggregate hashes, common launch geometry,
independently reconstructed loading and cluster intervals, quadrature,
sample counts and retention evidence. It writes `final/REPORT.md`,
`final/results.csv`, paired contrasts, `final/final_audit.json` and the
visually inspected PNG hashes in `final/visual_review.json`. No changes to
simulation inputs or retained results were necessary. Both plots passed
visual review. The heartbeat is stopped after final delivery.
