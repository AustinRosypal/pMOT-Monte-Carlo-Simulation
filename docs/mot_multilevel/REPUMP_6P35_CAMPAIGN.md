# Two loading calculations with a 6.35 mm repumper

## User request and order of work, 2026-10-02

1. Independently simulate cooling diameter 6.35 mm and 12.7 mm, both with
   repump diameter 6.35 mm; report both loading rates and uncertainties.
2. Add the matched 6.35 mm point to the loading-versus-cooling-diameter curve.
3. After those simulations and plot QA, answer which powers and detunings are
   best supported by prior corrected-model relationship results when both
   diameters are 12.7 mm. Give the supported loading value and compare with
   the conventional 27 mW cooling / 0.1 mW repump per traveling component,
   -15 MHz cooling detuning, and zero repump detuning.

The continuing convention is constant peak intensity, communicated to the
user before launch. Each of six repump components has 0.025 mW at 6.35 mm.
The six cooling components each have 6.75 mW at 6.35 mm or 27 mW at 12.7 mm.
Diameters are Gaussian 1/e^2. Cooling detuning stays -15 MHz; repump detuning
stays zero. Sampling radius and launch-plane distance are both 15 mm.
Directions are uniform over the full sphere; points are uniform in disc area.
Gravity, the 10 G/cm axial gradient, and all remaining physical/numerical
settings follow the completed diameter campaigns. Temperature and 2/3 mm
diameter cases remain deferred.

## Model scope

Use only the corrected Section-12 24-state population-rate equations under
`pmot.mot_multilevel`, with W=Gamma_e|Omega|^2/(Gamma_e^2+4Delta^2) and force
from W(p_g-p_e). The source digest remains
`5ac7e8791ace6010d2311b93443a3d3f9f0601e84f5a4ab38f653827a07904e5`.
This task continues the local-steady-state mean-force model, with no recoil
diffusion. Numerical qualification does not resolve the separate applicability
of instantaneous populations in weak repump wings; carry that limitation into
the final interpretation and power/detuning recommendation. The mechanism
diagnostics and transient examples are recorded in
`MATCHED_REPUMP_DIAMETER_CAMPAIGN.md`. Do not silently change to time-dependent
population dynamics during this comparison.

## Campaign files and process discipline

Runner: `scripts/run_repump_6p35_campaign.py`.
Statistics root:
`outputs/statistics/mot_multilevel_population_rate_v1/repump6p35_two_cooling_20261002_fixed_peak_intensities`.
Figures use the same campaign name under `outputs/figures/mot_multilevel_population_rate_v1`.
Prior matched and fixed-repump data and plots remain unchanged. Updated plots
are distinct files named `loading_rate_matched_with_6p35` and
`loading_rate_comparison_with_6p35`, plus analogous loading-coefficient plots,
in PNG and PDF. The 12.7 mm cooling / 6.35 mm repump point is a separately
labeled marker in the comparison, never a member of the matched-diameter curve.

Inspect `process.json`, `launcher.json`, live PID commands and worker parents,
and any queued helper before launch or resume. Never duplicate pools. Use one
16-worker pool, one BLAS thread per worker, and run from the repository root:

`uv run python -u scripts/run_repump_6p35_campaign.py all --workers 16`

Background launch uses a hidden window and redirects to `run.log` and
`run.stderr.log`. The manifest pins runner, helper, adapter and core hashes.
Changes require recorded revisions and revalidation; never bypass hashes or
edit live source. Full parameters and per-case compiled arrays are saved.
Read `failure.json`, `production_issues.json`, `pilot_progress.json`,
`sampling_plan.json`, `progress.json`, and `completion.json` when present.

## Sampling and gates

Each pilot uses 32 direction discs by 64 points. Case 0 (6.35 mm cooling)
uses pilot seed 2026100201 and production seed 2026100202. Case 1 (12.7 mm
cooling) uses 2026100211 and 2026100212 respectively. Cases and production
versus pilot therefore use independent random samples.

Production uses 16 points per disc, at least 64 discs, and pilot estimates of
between-disc and within-disc variance to choose fixed allocations. The target
is <=5% relative half-width of the pointwise 95% Student-t disc-cluster interval,
with a 50% variance safety margin. Freeze both allocations before production.
Do not assume a sample count establishes precision. If precision fails,
retain the run and use a justified independent fixed-size confirmation rather
than repeatedly adding samples until it passes.

Local geometry/force validation passed 64 comparisons with high-precision
references in weak wings. Fresh workers compare adaptive and fixed-RK4 results
at 32 nodes across the two cases. Full fixed-RK4 comparison checks all 328
speed nodes on three predetermined rays per case before production. The
unchanged dual-setting integrator, fixed edge checks, timestep/duration
fallback ladder, nonmonotone masks, tail gates, <=2% nested quadrature gate,
and representative longer-retention audits remain required. Timeouts remain
indeterminate and are never counted as escapes. Retain every attempted node.

Pilot results supply per-case runtime estimates. After production, recompute
loading and cluster intervals from saved disc spectra and verify retained-ray
counts and retention evidence; `aggregate_audit.json` records this audit.
Keep historical vapor normalization unchanged and report the density-independent
loading coefficient R/n87 as well.

## Completion and subsequent prior-results review

After both cases qualify, inspect all four PNGs and save a final report with
the two loading values, error bars, sample counts, powers, measured runtime,
numerical uncertainty, source provenance and plot links. Set visual review
complete only after inspection and retain figure hashes.

Then read the completed corrected population-rate relationship data under
`outputs/statistics/mot_multilevel_population_rate_v1/relationships_20260919`
and its final audit/revision reports. Recover actual parameter values from
metadata, not archived `mot_error` predictions. Identify the best measured
conditions on each scanned slice and the corresponding loading intervals.
Check whether repump power/detuning were ever varied. Do not claim a joint
four-parameter optimum, combine independently optimal coordinates into an
unmeasured loading prediction, or call an end-of-grid maximum a proven optimum.
If prior data cannot establish an optimal repump or joint operating point,
state that directly while giving the best supported measured comparison.
Use both 12.7 mm beams, the same sampling radius and vapor convention when
comparing to the baseline. Save the review and answer the user's second
question after completing the requested simulation/plot task.

Keep the heartbeat active until BOTH tasks are done. Stay quiet during normal
collection; notify only for meaningful milestones, failures, needed decisions,
or completion. Preserve all prior results and stop the heartbeat at the end.

## Completed 2026-10-02

Simulation finished at 22:02:32 UTC after 2.703 hours including pilots and
validation. The 6.35/6.35 mm case gives (6.3607 +/- 0.2815) million atoms/s
from 1,856 discs x 16 points; the 12.7/6.35 mm case gives
(15.2771 +/- 0.6137) million atoms/s from 688 discs x 16 points. Bars are
pointwise 95% Student-t direction-disc intervals. All 4,838 representative
retention cases passed; quadrature sensitivities were 0.000832% and 0.05126%.
All four new PNGs were inspected and the original campaign figures preserved.

The separate `scripts/finalize_repump_6p35_campaign.py` rechecks source hashes,
retained sample counts, disc integrals, cluster intervals, nested quadrature
and retention evidence without modifying simulation inputs or results.
Its hashes, results, visual review and report are under this campaign's
`final/` directory. No solver or runner revision was needed.

The subsequent prior-results review is complete in
`final/OPERATING_POINT_REVIEW.md`: the best measured prior loading mean is
(91.9605 +/- 6.8593) million atoms/s at 27 mW cooling per component,
-22.7625 MHz cooling detuning, 0.1 mW repump per component and zero repump
detuning, with both diameters 12.7 mm. The same relationship campaign's exact
-15 MHz baseline is (61.6116 +/- 1.3398) million atoms/s. Repump power/detuning
were not swept, and no joint optimum or combined-setting prediction is claimed.
The local-steady-state applicability limitation remains explicit in both reports.
