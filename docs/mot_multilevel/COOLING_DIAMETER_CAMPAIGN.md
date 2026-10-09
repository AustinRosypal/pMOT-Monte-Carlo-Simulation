# Cooling-beam diameter campaign, September 30, 2026

## Completed October 1, 2026

The authorized 5--25 mm production campaign is complete. All eight settings
passed their statistical, velocity-grid and representative retention gates.
Production plus audits took 6.737 hours (07:05:54--13:50:08 UTC), excluding
development and pilot work. There were 110,336 rays, 6,896 independent discs,
and 12,924 passing longer-duration retention cases. Statistical 95% half-widths
range from 3.515% to 4.483%; all nested velocity-grid differences are below
0.070%. No independent confirmation was needed.

Final report/data/audit and recorded visual inspection are under
`final_from_5mm` in the statistics root. Both PNG and PDF figures are under
`from_5mm` in the matching figures root. The final audit recomputed loading
integrals and Student-t intervals, checked ray counts, source/runner hashes,
retention evidence and saved-figure hashes. Both PNGs were visually inspected.
`production_5mm_completion.json` records the original production finish time
and final QA completion; its file modification time is not the runtime clock.
The 2 and 3 mm settings remain deferred. Do not restart completed worker pools.

## Current authorized production scope (October 1)

The user now requests **5, 7, 10, 12.7, 15, 18, 21 and 25 mm**, retaining
fixed peak intensity and the established uniform-area sampling method.
The 2 and 3 mm runs are deferred. The earlier stratification question is
superseded; it does not block this production run and stratification is not
authorized or used. The user expects unattended collection over the next
eight hours; keep the single resumable pool progressing through the settings.

Active runner: `scripts/run_diameter_production_5mm.py` (16 workers).
Read `production_5mm_process.json`, `production_5mm.log`,
`production_5mm.stderr.log`, `progress.json`, and any
`production_5mm_issues.json`, `production_5mm_failure.json`, or
`production_5mm_completion.json`. Check the recorded PID AND live command
before starting a pool. Earlier collector/audit/validation processes finished.

`production_5mm_plan.json` freezes an independent seed-2026093002 production
sample of 110,336 rays, with 16 points/disc and respective disc counts
3,792; 1,408; 544; 320; 240; 208; 192; 192. The pilot projections are
8.01--11.21 hours total including representative retention audits and runtime
allowance. These are estimates, not an eight-hour completion promise.
All selected pilots passed retention and nested quadrature checks. The
20 preselected full fixed-RK4 comparisons passed all 6,560 nodes without a
classification difference. Physics and the original adapter remain unchanged;
scope and runner provenance are in `revision_history/04_production_from_5mm`.

Human-readable parameters: `PRODUCTION_PARAMETERS.md` in the campaign root.
Full configuration, beam vectors/polarizations and numerical settings:
`production_parameters.json`. ARC-derived compiled inputs are saved under
`production_parameters/`. Result rows are `production_5mm_summary.csv`;
figures are in the matching figures root's `from_5mm` subdirectory.

If one setting fails a qualification gate, the runner retains its evidence,
labels it unqualified, and continues the next independent setting. Inspect
issues during monitoring; do not promote unqualified rows to final results.
Preserve failed precision runs and use an independently seeded fixed-size
confirmation if required. Diagnose genuine retention failures using retained
node evidence and unchanged physics before deciding a revision. All source or
runner changes still require a recorded revision and appropriate validation.
When the eight points pass, inspect saved PNGs, write the final report with
sample sizes, measured times and both uncertainty types, mark visual review
complete, and stop the heartbeat. Historical sections below retain provenance.

User request: loading rate versus Gaussian cooling-beam 1/e^2 diameter at
2, 3, 5, 7, 10, 12.7, 15, 18, 21 and 25 mm. Sampling discs remain 15 mm in
radius. User confirmed fixed peak cooling intensity and -15 MHz detuning.
Cooling power per traveling component is 27*(D/12.7)^2 mW. Repump components
remain 12.7 mm diameter and 0.1 mW each. Gravity is enabled.

Only the corrected 24-state Section-12 population-rate model is used. The
geometry adapter constructs explicit cooling and repump waists, changes the
compiled optical waist and Rayleigh-length entries, and is checked against
the full 24-state reference. The old worker cache lacks diameter in its key,
so the adapter clears that cache before each immutable ray payload.

## Locations and execution

- Runner: `scripts/run_population_diameter_campaign.py`
- Statistics: `outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline`
- Figures: matching `outputs/figures/mot_multilevel_population_rate_v1` root.
- Environment: repository-local `.venv`, selected portably with `uv run`.
- Full run arguments: `scripts/run_population_diameter_campaign.py all --power-mode fixed_intensity --detuning baseline --workers 16`.

Before launching/resuming, read `process.json` and check the PID **and command**
against the live process table. Never start a duplicate worker pool. Check
`run.log`, `run.stderr.log`, `run_failure.json`, and `completion.json`.
The initial `fixed_power_baseline` root contains only superseded preflight
validation evidence, from before the user's power choice; it is not production.

## Numerical revision

The historical source digest was
`3ee6d94ba7c5d962ddb7e225a86c78ea6f52cd6707a59052d3ab0096491e6224`.
New narrow cooling beams expose very unequal F=1 repump and F=2 cooling rates
in their far wings. Both the ordinary reference matrix solve and the original
globally scaled Schur solve lost ground-population accuracy there.

The corrected Schur system is column-equilibrated before solving, then mapped
back to normalized ground plus excited populations. Branching factors are
multiplied by the stimulated rate last to avoid intermediate underflow. When
all five F=2 absorption columns are exactly zero, a normalized mixture of
these absorbing ground states solves the rate equations and the stationary
optical force is exactly zero. The trajectory still evolves under gravity;
this branch does not decide capture or escape and introduces no rate cutoff.

The new core SHA-256 is
`5ac7e8791ace6010d2311b93443a3d3f9f0601e84f5a4ab38f653827a07904e5`.
`revision_history/01_population_conditioning` retains the original source,
intermediate validation evidence and scientific justification. All historical
campaign results/manifests remain unchanged. Changes to this runner or the
core require a recorded new revision and renewed validation, not a silent
manifest bypass.

Validation includes 65 multilevel tests, 240 local force/population comparisons
across all ten diameters (400-digit full-matrix reference in narrow-beam wings),
event/100-ms trajectory checks, and full fixed-RK4 classification comparisons
on two predetermined pilot rays per diameter before production. The 12.7 mm
optical arrays must be exactly identical to the baseline preparation.

The first pilot attempt exposed a stale Numba disk cache: the adaptive caller
loaded an inlined older force solver despite the current Python source hash.
Its 2,048 error files, with no accepted results, are retained under
`revision_history/02_compiled_cache_isolation`. The runner now sets
`NUMBA_CACHE_DIR` to `outputs/compiled_cache/<core SHA>` before importing any
numerical modules. Fresh spawned workers must pass 80 adaptive/fixed-RK4
launch checks across all diameters before the pilot can run. The ordinary
source-tree compiled cache must not be used for this campaign.

## Statistics, timing and publication gates

### Active retention review (October 1)

The initial 2 mm pilot resolved all 2,048 rays but failed its representative
retention gate: disc 4, point 49 at 1 m/s met the early capture criterion and
subsequently escaped the 30 mm surface at both fixed timesteps. Its aggregate
loading estimate is unqualified; its nested velocity quadrature also exceeds
the 2% threshold. No production allocation has been frozen.

`scripts/audit_diameter_retention.py --index 0 --workers 16` checks all 3,693
early-positive nodes on 1,022 rays, retaining the unchanged early-capture
evidence separately. Results are under `retention_recheck/000`, with a separate
`process.json`, `progress.json`, `completion.json` and per-ray node records.
**Check this audit PID and command as well as the original runner PID before
starting any pool.** Its top-level logs are `retention_recheck.log` and
`retention_recheck.stderr.log`. The original runner is stopped at the failed
gate. The audit does not itself rewrite masks or authorize a loading claim.
Both definitive outward escapes can qualify a rejected capture; a duration
timeout remains indeterminate. Retained nodes require agreement within
10 micrometers and 25 ms final-core residence. See the preserved review record
under `revision_history/03_retention_gate_review`.

After the audit, inspect the complete node evidence before choosing revised
mask provenance, any finer velocity grid, and a statistically valid production
allocation. Do not simply remove the original failed gate or relaunch `all`.

The serial continuation helper `scripts/start_diameter_pilot_continuation.ps1`
waits for this audited PID to exit and checks its completion file, then starts
`scripts/collect_diameter_pilots.py --start-index 1 --workers 16`. It records
`pilot_continuation_launcher.json`. Check this queued launcher as well before
launching anything. The collector records `pilot_collection_process.json`,
`pilot_collection.log`, `pilot_collection.stderr.log`, and
`pilot_collection_progress.json`. It collects diameters 3--25 mm independently,
retaining and recording any failed retention gate while continuing the next
setting. It never authorizes production. Its per-point allocations are
explicitly provisional until all relevant numerical gates have been reviewed.
This scheduling change leaves source physics, launch sampling and masks intact;
the collector has its own manifest and checks the original runner/core hashes.

### Retention review result and pending sampling choice (06:06 UTC heartbeat)

The all-positive 2 mm audit completed: 3,674 nodes retained, 19 definitive
escapes, zero indeterminate nodes. `scripts/review_diameter_retention.py`
generated separate masks and full source-file hashes under
`retention_qualified_pilot/000`; original pilot masks remain unchanged.
After those 19 replacements, the nested quadrature difference is 1.603%,
below the 2% gate. The pilot's 95% statistical half-width is 138.6%, so it
cannot support a quantitative loading claim or a confident extrapolation.

The rough independent-uniform production sizing is 2,154,496 rays at 2 mm,
with 320--448 hours including all-positive retention audits at measured pilot
throughput. The 3 mm pilot projects 368,384 rays and 33--46 hours with its
representative retention audit. These are provisional projections from small
pilots, **not authorized production launches or promised completion times**.
The 3--21 mm pilot settings passed retention and nested quadrature checks.

An asynchronous user question is pending: allow area-weighted radial
stratification (oversample the central capture region, retain sampling of all
annuli through 15 mm, apply exact area weights), or retain uniform independent
points over the entire disc. **Do not change the sampling convention without
the answer.** The existing repository explicitly specifies uniform disc-area
sampling. No stratified trajectories or production allocations have been run.
The alternative would require a recorded design revision, independent pilot,
estimator validation, and fresh runtime/allocation measurements.

Meanwhile `scripts/start_diameter_validation.ps1` waits for the current
collector and then starts `scripts/validate_diameter_pilot_masks.py` to check
all 328 nodes on the original two preselected rays at every diameter with
fixed RK4. Check `fixed_validation_launcher.json`,
`diameter_fixed_validation_process.json`, `fixed_validation.log`, and
`fixed_validation.stderr.log` before starting another pool. This validates
the early-capture masks separately from the retention qualification and does
not authorize production. The helper verifies the immutable runner/core hashes.

The independent pilot uses 32 direction discs by 64 uniform-area points per
disc at each diameter (20,480 rays total), seed 2026093001. It estimates
between-direction and within-disc variance. Production uses independent seed
2026093002. Candidate allocations use 16, 32, 64 or 128 points per disc, at
least 64 independent direction discs, and a 50% variance safety margin.
Each fixed production allocation is frozen before its data are collected.

The loading target is a **95% Student-t direction-cluster half-width <=5%**
at every point, verified after production. A zero-capture pilot requires more
diagnostic sampling or an explicit upper-limit design, not a zero-error result.
If production misses precision, preserve it as sizing evidence and use an
independent fixed-size confirmation; do not repeatedly add samples until a
confidence interval happens to pass.

`pilot_progress.json` supplies per-diameter variance and runtime estimates as
each pilot point completes. `sampling_plan.json` freezes the final allocations
and estimated production time. Estimates use measured pool wall time and
retention-audit time, with 25--75% runtime allowance. These estimates exclude
already spent development/pilot time and are conditional on all gates passing.

Each ray uses the unchanged 328 positive speed nodes: 0.25--80 m/s by 0.25,
plus 85--120 m/s tail probes by 5. Preserve nonmonotone masks. Dual adaptive
integration, fixed RK4 edge checks and fallback timestep/duration ladders,
and representative fixed-RK4 longer-duration retention checks remain enabled.
Timeouts remain indeterminate. Upper-tail capture blocks quadrature until a
dense extension is implemented and validated. Nested quadrature difference
must be <=2% of the loading mean; statistical precision is checked separately.

The prescribed finite launch geometry remains fixed. This is loading within
the 15 mm sampled radius; it does not establish convergence to an unbounded
incident plane for the largest beams. Repump diameter is held fixed, so any
repump-limited behavior is part of this cooling-only scan.

The unresolved historical vapor normalization is retained and explicitly
labeled for continuity. Also report the normalized-Maxwell loading coefficient
R/n87 in m^3/s, with its confidence interval. Changing a density normalization
does not require rerunning trajectories. Temperature remains deferred.

At completion, inspect PNG figures, save a report with per-point counts,
statistical and numerical uncertainties, and source provenance; mark visual
review complete. Stop the campaign heartbeat only after successful final QA.
