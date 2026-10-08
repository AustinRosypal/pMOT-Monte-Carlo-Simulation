# Matched cooling and repump diameter campaign

## Authorized comparison, 2026-10-02

Repeat the completed cooling-diameter experiment with the repump diameter
equal to the cooling diameter. The user explicitly selected constant peak
repump intensity, as well as the existing constant peak cooling intensity.
Diameters are 5, 7, 10, 12.7, 15, 18, 21, and 25 mm (Gaussian 1/e^2).
The 2 and 3 mm points and temperature remain deferred.

At diameter D in mm, each of six cooling traveling components has
27*(D/12.7)^2 mW; each of six repump components has 0.1*(D/12.7)^2 mW.
Both families have waist radius D/2. Cooling detuning is -15 MHz and repump
detuning is zero. All other apparatus, force, trajectory, sampling, and vapor
normalization settings are inherited unchanged from the completed baseline.
The comparison changes the repump spatial profile while keeping its central
intensity fixed; it does not compare equal total repump powers.

## Files and execution

Runner: `scripts/run_matched_repump_diameter_campaign.py`.
Statistics root:
`outputs/statistics/mot_multilevel_population_rate_v1/matched_repump_diameter_20261002_fixed_peak_intensities`.
Figures root:
`outputs/figures/mot_multilevel_population_rate_v1/matched_repump_diameter_20261002_fixed_peak_intensities`.
Baseline root:
`outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline`.

Before starting or resuming, inspect `process.json`, `launcher.json`, and the
live PID command lines, including workers and queued helpers. Never duplicate
pools. The runner uses one 16-worker pool and one BLAS thread per worker.
Use the prescribed interpreter via
`wsl.exe --exec /home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -u scripts/run_matched_repump_diameter_campaign.py all --workers 16`.
Background launches use a hidden window, with stdout in `run.log` and stderr
in `run.stderr.log`. Check `failure.json`, `pilot_progress.json`, `progress.json`,
`sampling_plan.json`, `production_issues.json`, and `completion.json` as available.

The manifest pins the new runner, unchanged baseline adapter, and physics
source hashes. Current physics SHA256 is
`5ac7e8791ace6010d2311b93443a3d3f9f0601e84f5a4ab38f653827a07904e5`.
Never edit running source or silently bypass manifest checks. Changes require
a recorded revision, retained earlier evidence, and relevant revalidation.
The baseline outputs are immutable.

## Parameters and validation

`parameters.json` records full settings, SI constants, coil geometry/current,
beam polarizations, powers, diameters, wavelengths, numerical controls,
capture definition, sampling conventions, and normalization. Per-diameter
compiled inputs are saved under `parameters/`.

The corrected 24-state population-rate solver remains the only physics engine:
W = Gamma_e*|Omega|^2/(Gamma_e^2+4*Delta^2) and beam force from W*(p_g-p_e).
Gravity is (0,0,-9.80665) m/s^2. Recoil diffusion is disabled. The axial
quadrupole gradient is 10 G/cm. Cooling peak intensity is 42.628145214 mW/cm^2;
the repump reference is 0.1 mW at 12.7 mm. Propagation-frame helicities on x,y,z
are sigma+, sigma+, sigma- for both traveling directions and both families.
The radiation force and populations are recomputed along trajectories.

Local validation passed 192 full-population/force comparisons, including
16 high-precision wing checks, constant peak intensity checks for all beams,
and exact compiled-array identity with the baseline at 12.7 mm. The prior
unchanged-kernel test and event/trajectory certificates are checked by hash.
Before pilots, fresh spawned workers compare adaptive and fixed-RK4 outcomes
at 64 nodes across all eight diameters. Before production, compare all 328
nodes on two predetermined pilot rays per diameter with fixed RK4, retaining
the evaluation evidence. Any mismatch blocks production.

## Sampling and uncertainty

Sampling radius and launch-plane distance are both 15 mm. Directions are
uniform over the full sphere, with uniform-area points on each perpendicular
disc. Velocities are parallel to its normal toward the MOT. No 4*pi or octant
multiplier is used. The finite sampling radius is held fixed by instruction;
this scan does not establish convergence to an unbounded launch plane.

Pilot seed 2026093001 uses 32 discs by 64 points for every diameter. Production
seed 2026093002 is independent of the pilot and matches the baseline geometry.
Production uses 16 points per disc. Each allocation is at least the baseline
disc count (3792, 1408, 544, 320, 240, 208, 192, 192 respectively), and is
increased if pilot between/within-disc variance requires it. This gives a
minimum of 110336 production rays, not a claim of adequate precision.
Allocations include a 50% variance safety margin and are frozen before any
production results are collected. Target pointwise 95% Student-t direction-disc
cluster relative half-width is <=5%. If it fails, retain the result and design
an independent fixed-size confirmation; do not keep adding data until it passes.

`pilot_progress.json` records measured per-diameter runtime estimates;
`sampling_plan.json` freezes the total estimate and allocations. Wall-time
estimates include observed capture and retention work and 25--75% allowance.
Report them after the pilot, rather than assuming baseline timing applies.

The established dual-integrator and fixed-step/duration fallback ladders,
328 positive speed nodes, nonmonotone masks, representative longer retention
audits, and <=2% nested quadrature sensitivity gate remain unchanged. Timeouts
are indeterminate, never escapes. Every evaluated node is retained. A failed
retention or numerical gate withholds qualification even when statistical
precision passes. Review the evidence and continue other independent settings
where supported; never ignore a failed gate.

Main-curve error bars use all production discs per point. `paired_comparison.csv`
uses the first baseline-count discs from the new run and all baseline discs.
The runner verifies the launch arrays are identical before reporting Student-t
intervals of per-disc loading differences. These paired contrasts isolate the
repump change with less Monte Carlo noise. The 12.7 mm paired difference must
be exactly zero because geometry and physics are identical. These are pointwise,
not simultaneous confidence intervals.

## Outputs and final review

Save loading-rate and normalized-Maxwell loading-coefficient plots, both alone
and overlaid on the previous fixed-12.7-mm-repump curve. Preserve the historical
vapor prefactor unchanged; also report R/n87 with n87=5.451556957e12 m^-3 for
its equivalent normalized Maxwell distribution. Statistical intervals do not
include uncertainty in vapor density, apparatus parameters, or the model.

When complete, inspect every saved PNG, check all eight statistical and
numerical qualifications, and save a report with counts, measured timing,
uncertainties, paired repump-size differences, and provenance. Mark visual
review explicitly complete. Stop the heartbeat only after final QA and report.
While the process advances normally, remain quiet. Notify only for meaningful
stage completion (including measured pilot ETA), failure, needed user decisions,
or final completion.

## Completed and reviewed, 2026-10-02

Production finished at 14:36:41 UTC. All eight points passed; no independent
confirmation or sample extension was needed. The final allocation was 110336
rays on 6896 discs, with 16 uniform-area points per disc. Statistical 95%
relative half-widths range from 2.896% to 4.318%; maximum nested quadrature
sensitivity is 0.0741%. All 12868 representative retention cases passed.
Production plus retention audits took 6.897 hours; total execution including
pilots and preproduction validation took 7.746 hours.

`scripts/finalize_matched_repump_diameter.py` independently recomputed disc
integrals, Student-t intervals, paired differences, and retention metrics.
Its hash and all input/figure hashes are recorded in `final/audit.json`.
All four PNGs were visually inspected; `final/visual_review.json` pins them
by hash. `completion.json` records successful final review. The report and
per-point timing, coefficients, power values, and paired intervals are in
`final/REPORT.md`, `final/results.csv`, and `paired_comparison.csv`.

Relative to the fixed-12.7-mm repump baseline, matching the repump changes
loading by -19.53%, -18.34%, -12.35%, 0%, +19.88%, +55.02%, +90.70%, and
+126.49% at the eight increasing diameters. Every non-reference point's
paired difference interval excludes zero; the 12.7 mm result is exactly
identical disc by disc. The 25 mm rate is (239.923 +/- 7.962) million atoms/s
under the retained historical vapor normalization. The comparison preserves
peak intensity, so both repump coverage and total repump power increase.

## Mechanism analysis and separate steady-state applicability issue

The user's October 2 mechanism question prompted post-run diagnostics in
`analysis_repump_mechanism/`, generated by
`scripts/analyze_matched_repump_mechanism.py` and
`scripts/plot_repump_mechanism.py`. These did not change production data or
physics. Script/source hashes and scope are recorded with the diagnostics.
The saved `mechanism.png` was visually inspected: the left panel contains
deterministic local population snapshots, and the right panel shows production
cross sections with 95% disc-cluster intervals.

For 25 mm cooling, on the (1,1,1) radial line at 12 mm from the center and
20 m/s inward speed, the local steady F=1 population is 77.28% with the
12.7 mm repump and 9.08% with the 25 mm repump. Cooling-light braking
accelerations are 10173 and 40717 m/s^2; direct repump braking is only
0.84 and 3.34 m/s^2. Thus the large force change here acts through cooling
populations, rather than direct repump momentum. At 15 mm radial distance,
the respective steady F=1 populations are 97.47% and 13.39%.

The saved 25 mm production cross sections at 30 m/s are 247.12 versus
587.21 mm^2, and at 35 m/s are 51.08 versus 311.55 mm^2. About 86.49% of
the net loading gain comes from launches with impact parameter greater than
6.35 mm. Impact parameter is relative to the incident direction, not a hard
repump boundary; the six Gaussian beam axes and changing trajectories matter.
Loading weights the cross section by v^3 exp(-v^2/A), which amplifies improved
capture of faster atoms. Near 5 m/s, both masks fill the prescribed 15 mm
sampling disc, so this remains a finite-disc comparison.

**Separate physical-model limitation:** these production trajectories use
instantaneous local steady-state internal populations, not dp/dt=M(r,v)p.
The numerical convergence and representative retention gates test external
dynamics under that assumption; they do not establish its applicability to
all incoming atoms in weak repump wings. Frozen-environment checks illustrate
the concern. At 15 mm on the same diagonal and 20 m/s, starting with
unpolarized F=2, the 12.7 mm repump gives only 18.55% F=1 after 50 microseconds,
whereas local steady state gives 97.47%. The atom travels 1 mm in that time.
The matched 25 mm case reaches 10.38% versus a 13.39% steady value. These are
constant-environment transients, not alternative loading trajectories, and
their result depends on the specified initial internal state.

The qualitative repump-coverage mechanism is directly demonstrated within
the model. The precise 126.49% loading enhancement should remain conditional
on the steady-state approximation until time-dependent population trajectories
with documented incident populations are compared on representative rays
whose classifications differ between the two repump geometries. No such
coupled internal/external capture validation was performed in this analysis.
