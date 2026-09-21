# Corrected population-rate relationship campaign

Requested 2026-09-19. Temperature and its Doppler overlay were explicitly
deferred by the user. This campaign produces the other seven requested plots.
No runtime imports from `mot_error` are permitted. Historical files supply
scan definitions only, never forces, rates, populations, or trajectories.

## Completed September 21

All seven requested plot types and all audits are complete. Final selected
results contain 483,072 rays at 75 loading settings, plus 111 force detunings
on three axes. Final statistics, report, provenance and visual QA are under
`outputs/statistics/mot_multilevel_population_rate_v1/relationships_20260919/final`;
PNG/PDF figures are under the matching `outputs/figures` root.
All 60,073 representative retention checks passed. Loading 95% half-widths
are 0.94--9.53%; the largest nested velocity-quadrature difference is 0.207%.
The confirmation pool exited normally; no simulation workers need restarting.
Original failed-precision summaries remain unchanged for provenance.
Temperature remains deferred. The heartbeat can now be stopped.

## Historical stage record: independent confirmation (September 21)

The original 75-setting production campaign completed all 306,944 rays and
all numerical/retention checks. Seven settings missed the loading precision
target. The original process has exited; do not restart that completed pool.
The active 16-worker pool now uses
`scripts/run_population_precision_confirmation.py --workers 16`, with its own
`process.json`, `progress.json`, `run.log`, `run.stderr.log`, and completion
under `outputs/statistics/mot_multilevel_population_rate_v1/relationships_20260919_precision_confirmation_01`.
Read that root when monitoring. `active_campaign.json` in the original root
also points to it. Always check the saved PID and full command before any launch.

The final frozen confirmation plan has seven settings / 239,616 rays:
raw saturation 0.25: 3184 discs; 0.5: 1104 discs; detuning -5.0: 880 discs;
-5.25: 1296 discs; -5.5: 1920 discs; -5.75: 2720 discs; -6.0: 3872 discs.
Every disc has 16 points. All original high-contribution direction checks and
retention audits passed. At -6.0, the original loading half-width was 21.21%.
The confirmation uses independent seed 2026092001 and fixed counts, preserves
the original samples, and excludes the selected originals from replacement
estimates. Do not edit the frozen plan after confirmation starts. Recheck its
precision and numerical gates at completion; then assemble final figures with
per-point source-root provenance. The measured estimate is roughly 10--12
hours from the September 21 ~00:35 CDT confirmation launch, subject to gates.
The heartbeat remains active until all seven final figures and audits pass.

September 21, 04:34 CDT: both raw-saturation confirmations passed (95%
loading half-widths 8.12% and 7.56%). The complete 24-point raw-saturation
figure is now validated, with 114,432 selected rays and 6.09--8.60% intervals.
Six of seven figure types are assembled under the original figure root's
`final` subdirectory; the detuning confirmations remain active. Preserve the
original provisional figures and datasets.

The postprocessing-only launcher
`scripts/assemble_population_relationship_results.py --available` audits
source hashes, seeds, frozen confirmation allocations, retention checks, and
recomputed direction-cluster loading intervals. It writes only fully validated
studies, plus per-point provenance in the original statistics root's
`final/assembly_manifest.json`. Its source hash is recorded in that manifest;
it does not alter the live simulation. Once all seven confirmation settings
pass, run the same script without `--available` to require all 75 authoritative
points and all seven primary figures. Inspect those saved figures and the
supplementary full-speed cross-section plot before final delivery. Finish the
scientific/sample-size/uncertainty report and stop the heartbeat only then.

## Current runtime policy (revision 4; reporting fix in revision 5)

The user rejected the roughly 13-day estimate and requested approximately
26-hour execution. The original 12-worker pool was stopped after verifying its
PID and command; all checkpoints and prior manifests remain preserved.
`revision_history/04_runtime` records this revision, benchmarks, and validation.
The allocation and integrator descriptions below revision 4 are historical.

The revised method keeps every one of the 328 positive velocity nodes per ray.
It uses the exact Section-12 force with embedded Dormand--Prince integration
at two tolerances and two step caps. Every capture-edge neighbourhood is
rechecked using the original dual-timestep RK4 escalation ladder, including
new adjacent edges revealed by those checks. Adaptive disagreements also use
that ladder. Timeouts remain indeterminate. Per-evaluation method codes
distinguish fixed RK4 timesteps from adaptive maximum steps in saved records.
The longer-duration retention audits still use fixed RK4 and the original
endpoint/core criteria. No force interpolation or monotone threshold shortcut
is introduced.

The independent pilot determines a fixed revised allocation of 16 points per
disc and at least 64 independent direction discs, in increments of 16. All
eight radii use the largest required radius disc count and the same normalized
geometry. Loading-only sweeps need the unchanged 10% loading interval target;
the unchanged 5%-of-peak cross-section target applies to the eight requested
radius spectra. The 50% variance safety margin is retained. Previous fixed-RK4
results at selected indices are reused. Extra previously computed points are
preserved but are not silently mixed into the balanced revised estimator.
Production confidence intervals and all precision gates still must pass.

Before resuming production, require all tests and the exact-source hybrid
pilot comparison gate to pass. The comparison covers four dispersed rays at
every setting and every nonmonotone pilot ray (506 rays / 165,968 speed nodes).
The raw adaptive benchmark exposed six edge discrepancies; the hybrid must
recover the saved fixed-RK4 classifications at all selected nodes. Do not
launch production while a benchmark or validation process remains active.

Revision 4 passed: all 165,968 hybrid comparison nodes match, all 68 tests
pass, and the event/100-ms trajectory validation passed again. Source digest:
`a8412df03f16e5f55956d49784c1746a76b2c9091d6a4b392cd737a9db242429`.
The finalized allocation is 306,944 rays, including 992 discs x 16 points at
each radius. Production resumes with `production --workers 16`. The measured
16-worker capture projection is 11.67 hours, with a 16--24-hour total estimate
allowing for fixed-RK4 retention audits, processing, and runtime variation.
This is an estimate, not permission to relax a failed convergence or precision
gate. Reassess the estimate using actual production throughput.

Reporting revision 5 (September 20): the first radius completed all 15,872
rays and passed retention, loading/cross-section precision, and quadrature.
Plotting then exposed an all-captured mean 5.83e-19 m^2 above the disc area
from floating-point summation, producing a negative error-bar magnitude.
Boundary detection now uses the exact binary masks, pins full/empty means to
their physical boundary, and applies the existing nonzero boundary interval.
Forces, trajectories, allocation, and loading integrals are unchanged; the
original evidence is preserved under `revision_history/05_cross_section_bounds`.
The source digest is now
`3ee6d94ba7c5d962ddb7e225a86c78ea6f52cd6707a59052d3ab0096491e6224`.
The regression suite has 69 tests. The hybrid comparison is rerun for this
revision, and its exact-source certificate now lives at
`validation/hybrid_pilot_comparison.json`. Resume with 16 workers only after
the tests, hybrid certificate, event/trajectory checks, and figure regeneration
all pass. Archive the failure log before restart; never rerun a live pool.

## Pending independent precision confirmation

September 20, 18:25 CDT: all eight radius settings are validated. The first
raw-saturation settings, s0=0.25 and 0.5, passed trajectory/retention and
quadrature checks but their loading 95% half-widths are 20.38% and 10.67%.
They are explicitly unqualified against the unchanged 10% target. Inspection
of the three highest-contributing direction discs at each setting verified
resolved masks and fixed-RK4 agreement at every captured upper boundary.
Direction-dependent capture produces more variance than the small pilot
predicted; timeouts are not being counted as escape.

`precision_followup_plan.json` preallocates independent confirmation using
seed 2026092001: 3184 x 16 and 1104 x 16 rays (68,608 total), from the observed
direction-cluster variance and the same 50% variance safety margin. Original
failed-precision samples serve only as allocation evidence; they must not be
pooled into the confirmation estimates. Counts are fixed before confirmation
starts, with no early stopping when an interval looks small enough. Preserve
the originals and report the independent confirmation as replacement evidence.

September 20, 23:30 CDT update: detuning -5.0 and -5.25 linewidth also passed
all numerical/retention checks but missed loading precision (10.52%, 12.60%).
Their highest-contributing three direction discs have definitive fixed-RK4
capture boundaries. The saved plan now additionally allocates 880 x 16 and
1296 x 16 independent confirmation rays, respectively: four settings and
103,424 confirmation rays total. The plan's previous version is preserved in
`precision_followup_history`. These two additions are projected to take about
1.5 hours including audits; the four-setting confirmation is about 5 hours.
The launcher preflight passes for all four. The original pool remains active;
do not launch the confirmation pool until the original pool exits.

Continue the current pool uninterrupted. Inspect and append any further
precision failures to this plan before sampling those settings. Once all 75
original settings finish, check the saved PID and command and confirm no
campaign pool remains, then run the prepared launcher:

```powershell
wsl.exe --exec /home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python scripts/run_population_precision_confirmation.py --check
wsl.exe --exec /home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python scripts/run_population_precision_confirmation.py --workers 16
```

Use a hidden process and separate logs under the new
`relationships_20260919_precision_confirmation_01` statistics root. The runner
refuses to start before original completion and saves a separate source/seed/
allocation/runner manifest. `precision_followup_preflight.json` records its
hash and read-only preflight. It invokes unchanged, validated capture and
retention functions; no source or manifest change to the live campaign is
needed. Read the confirmation root's process, progress, logs, and completion
after it starts. When its gates pass, assemble the seven final figures from
the original validated settings plus these independently validated replacements,
recording source-root provenance for every point. Do not stop the heartbeat
merely because the original campaign reports all 75 points computed.

## Historical coordinates

Use the latest September refinement, not the superseded August grids:

- Disc radius: 3, 5, 8, 12, 15, 20, 25, 30 mm.
- Single-beam center raw saturation: 0.25, 0.5, 0.75, 1, 2, 3, 5, 10,
  15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 125.
- Effective saturation: 0.25 through 5 by 0.25.
- Loading detuning: -0.5 through -6 by -0.25 linewidth.
- Force detuning: -0.5 through -6 by -0.05 linewidth (111 points).
- Historical plotting-coordinate linewidth: 2 pi times 6.07 MHz;
  historical saturation reference intensity: 16.69 W/m^2. These define the
  requested scan coordinates only. Actual excited-state decay rates and
  dipoles remain the ARC-derived values in the corrected model.
- Saturation scans change cooling power only, at -15 MHz; other scans use
  27 mW per cooling traveling beam. Repump power is 0.1 mW per component.
- Loading relationship discs have 15 mm radius; disc centers are 15 mm
  from the origin. Radius studies reuse normalized geometry across radii.

## Statistical and numerical policy

Full-sphere isotropic directions, uniform-area disc points, and parallel
velocities within each disc. Direction discs are the independent clusters.
Report 95% Student-t pointwise intervals across discs; never count individual
trajectories as independent direction samples. A pilot determines the final
sample size before the independent production seed is used. Production
precision is checked and reported; a chosen sample count alone does not
establish sufficiency. Zero observed capture is not evidence of exactly zero
cross section.

Every capture node must pass a dual-timestep classification check, with
timeouts retained as indeterminate and escalated in observation duration.
Capture-speed masks retain nonmonotone islands. Longer trajectories must
also test the cheap core-residence/two-entry criterion. No force or capture
result is certified by a syntax check or by regression to archived results.

The independent pilot uses 16 discs with 16 points per disc at nine settings
covering small/large radii, weak/strong light, and the detuning range. It
selects at least 64 production discs with 64 points per disc, using a 50%
variance safety margin. Targets are a 10% relative 95% loading interval
half-width and a cross-section interval half-width below 5% of the spectrum
peak over its main support. Any missed production target remains explicitly
unqualified. The seeds are 2026091901 (pilot) and 2026091902 (production).

Allocation revision 3 (2026-09-20): those nine settings completed with all
positive-speed nodes resolved and all duration audits passed. The initial
allocation formula exceeded its 256-disc guard because it scaled only disc
count and did not remove the within-disc contribution before projecting from
16 pilot points to 64 production points. The allocation now estimates
`Var(mean) = (between_disc_variance + within_disc_variance/P) / N` from the
per-ray pilot results. The existing nine-setting pilot is retained, and the
same 16 x 16 pilot is extended to every one of the 75 settings rather than
extrapolating variance between regimes. Production uses 64 points per disc
and a per-setting number of discs in multiples of 64, with the original 50%
variance margin and unchanged precision targets. All eight radius settings
share their largest required disc count to preserve paired geometry. The
4096-disc allocation guard triggers scientific review rather than silently
weakening a precision requirement. Saved revision records prove that the
physics files, capture/classification/statistics functions, and original
pilot evidence did not change. Production remains on its independent seed.
Revision 3 revalidation passed 59 tests and the event/trajectory checks.
The resumed `pilot --workers 12` stage computes the remaining pilot settings
and writes the complete `sampling_plan.json`. On its successful exit, continue
with `production --workers 12` (using the same launcher and prescribed
interpreter); never rerun a live pool. A pilot-stage exit without a complete
sampling plan is a failure to investigate, not permission to guess counts.

The 75-setting pilot completed on 2026-09-20: 19,200 rays, all node,
retention, and velocity-quadrature gates passed. It identified 235
nonmonotone rays. `pilot_completion.json` records the source and allocation
digests. The independent production plan contains 819,200 rays: 640 discs
x 64 points at every radius, and 64--448 discs x 64 points at other settings.
The exact per-setting counts are in `sampling_plan.json`. Production started
with 12 workers using `production --workers 12`; it is about 43 times the
pilot workload. Precision must still be verified from production data.
The completed pilot logs are preserved as `pilot_run.log` and
`pilot_run.stderr.log`; canonical `run.log` now follows production.

Direct masks use 0.25--80 m/s in 0.25 m/s increments. Additional 85--120 m/s
probes check the high-speed tail. Capture in the upper 5 m/s of the dense
grid or any tail probe blocks aggregation pending a domain extension.
The radius figure retains the historical 1--30 m/s speed coordinates.
No sigma(0) is imputed: the loading integral uses the exact g(0)=0 anchor.
Every direction disc with capture contributes an outermost captured ray to
the longer-duration audit, at its highest and median captured grid speeds.

Deterministic restoring slopes and damping extrema receive numerical
resolution error bars, not fictitious Monte Carlo confidence intervals.
Restoring slope means dF_i/dx_i at zero velocity. Damping turnaround means
the positive speed where F_i(v_i) is most negative, not a force zero.
Radiation-force diagnostics exclude gravity; capture trajectories include it.

All new results, checkpoints, audit records, manifests, and figures go under
`outputs/<category>/mot_multilevel_population_rate_v1/relationships_20260919`.
Save the source digest, model identity, package versions, exact grids,
geometry, numerical controls, and individual evaluated nodes. Resume only
matching configurations and code. Never overwrite historical products.

## Execution notes

The prescribed interpreter currently points through WSL to a Windows Python
runtime. Campaign dependencies are installed into the ignored project-local
`.venv_pMOT_MC/Lib/site-packages`, without changing the shared runtime.
Launchers bootstrap that directory and `src` before importing project code.
Every Python invocation still uses the prescribed interpreter path.

Launch/resume with:

```powershell
wsl.exe --exec /home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python scripts/run_population_relationships.py all --workers 12
```

Stages `validate`, `force`, `pilot`, `production`, and `report` are available.
Do not start a second worker pool while the existing process is active.
`campaign_manifest.json` locks numerical controls and the source digest;
`progress.json`, `process.json`, and the run logs describe the active work.
`run_failure.json` records an exception, and per-ray NPZ files retain every
evaluated node for diagnosis. The manifest check intentionally rejects a
resume after a physics/numerics change; use a reviewed new campaign revision.

Validation completed before the initial launch: 56 tests; compiled/reference field,
population, force, and RK4 checks; four frozen-point Gillespie comparisons;
five 100 ms dual-timestep retention checks. These precede, and do not replace,
the per-node and per-setting production convergence audits.

The first startup attempt was stopped after exposing concurrent ARC SQLite
initialization on Windows. Its complete statistics and figures are retained
in sibling `relationships_20260919_startup_attempt_01` directories. Workers
now receive the parent's precomputed atomic arrays and never construct ARC.
Checkpoint replacement also retries transient Windows/Synology locks. The
physics kernel, geometry, velocity grid, and numerical tolerances did not
change. The fresh run revalidates and recomputes the pilot; no failed-startup
ray is silently reused. A regression check guards the no-ARC-in-workers rule.
