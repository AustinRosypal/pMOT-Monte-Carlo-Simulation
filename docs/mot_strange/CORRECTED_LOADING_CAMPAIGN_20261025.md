# Corrected mot_strange loading campaign, 2026-10-25

The user authorized starting this simulation and periodic calculated loading
rate/uncertainty updates. Keep the historical numerical vapor normalization,
explicitly interpreted with a normalized Maxwell distribution: T=296.17928934 K
(23.02928934 C), n87=5.45155695704e12 m^-3. Preserve both values with every rate.
Also preserve the per-disc velocity spectra, masks, and R/n87 for later
temperature/density reweighting. These are assumed vapor conditions, not a
new density measurement.

Runner: `scripts/run_corrected_mot_strange_loading.py`.
Command: `uv run python -u scripts/run_corrected_mot_strange_loading.py all --workers 16`.
Output root:
`outputs/statistics/mot_strange/calibrated_10_10_20Gpcm_full_sphere_r15mm_incell_20261025`.

The calibrated signed origin gradients are (+10,+10,-20) G/cm. Surrogate path
powers are (21.7191389574,21.7191389574,43.4382779148) mW per traveling
component, using the corrected kappa=-9.740755511403248e-8 T/(W/m^2) and the
existing center-balanced 1530 nm surrogate geometry. Cooling remains 27 mW per
component at -15 MHz, repump 0.1 mW per component at zero detuning; both beam
diameters are 12.7 mm. Gravity is enabled, recoil diffusion disabled.

Full-sphere direction discs have 15 mm radius and centers 10 mm from the
origin. Reject outside-cell proposals until every disc has 20 accepted
uniform-area points. Never reject directions. All velocities on a disc are
parallel to the normal. The cell is the existing 28.2842712 mm-side cube.
Multiply each conditional capture fraction by its own disc/cell intersection
area, not the full nominal circular area. Save those areas and all geometry.

The physical Section-12 force kernel is reused through its explicit-field
compiled entry point. No independent scattering closure or force
interpolation is permitted. Every adaptive/RK4 stage reevaluates field,
intensity, polarization, detunings, rates, populations and net photon force.
The surrogate environment and wall termination live under mot_strange.

An independent 32-disc x20-point pilot (seed 2026102501) freezes a production
allocation, minimum 200 discs, rounded to 25, with a 1.5 variance safety factor.
Production seed is 2026102502. Target: 95% Student-t disc-cluster half-width
<=5% of mean loading. Interim intervals are descriptive; they do not determine
stopping. Never call an interim rate final. A missed target needs a separately
planned independent fixed-size confirmation, not optional stopping.

Every ray evaluates direct positive-speed nodes 0.25--80 m/s by 0.25 m/s plus
85--120 m/s tail probes. Every node requires two definitive agreeing numerical
settings; all capture edges and disagreements receive dual-fixed-RK4 checks.
Retain the fixed duration/timestep escalation ladder through 2 s. Literal zero
speed is absent from sigma; only the exact weighted-integrand zero-flux anchor
is used. Preserve nonmonotone masks and all attempted nodes. Do not relabel
timeouts, off-center nonterminal orbits, exceptions or unfinished work as
escape. Any required indeterminate node withholds a final rate.

Gates include source-matched core/accelerated/reference tests, local
reference-force and frozen-point event comparisons, fresh spawned-worker
adaptive/fixed comparisons, full fixed-grid comparisons on two predetermined
pilot rays, nested velocity quadrature <=2%, and longer dual-timestep core
retention checks on challenging captured rays from every disc. The central
core uses the unchanged 2 mm / 5 ms continuous-residence OR two-entry rule.
Off-center retained trajectories must be reported as separate diagnostics;
they do not silently count as central-core loading.

The runner currently computes interior-launch-aperture loading. Before
claiming total-cell loading, complete a separately documented aperture and
launch-placement convergence audit. The runner explicitly leaves this gate
pending. Also complete the final figure visual review and record hashes.
Do not mark the overall work complete while any convergence/precision/visual
gate remains pending. Frozen-point event comparisons do not validate moving
population adiabaticity in weak light; retain this model limitation.

Inspect `process.json`, `launcher.json`, actual live process commands/parents,
`run.log`, `run.stderr.log`, `progress.json`, stage `latest_estimate.json`,
`sampling_plan.json`, `validation`, `retention_progress.json`, `failure.json`
and `completion.json`. The runner has an OS file lock. Never duplicate worker
pools, change active source files, or stop unrelated laboratory/Jupyter jobs.
Source changes require a documented fresh revision and matched validation;
preserve old outputs. Interrupted identical-source work can resume from
retained ray nodes.

The user requested periodic values: report updated provisional rate, 95%
interval, number of complete independent discs, stage, and outstanding gates
when more complete-disc evidence becomes available. Do not send repeated
unchanged estimates or invent rates before complete velocity masks exist.
Notify on failure or a required scientific decision. At completion report
the rate, R/n87, thermal inputs, sample counts, timing, numerical sensitivity
and scope, with plots/report links; pause the monitor only when finished.

## Pilot gate failure and diagnostic continuation

The original collector stopped on 2026-10-25 at 20:20:26 UTC. Its immutable
output contains 500 resolved rays and four rays that stopped at an
indeterminate positive-speed node; 136 pilot rays were not started. The first
failure was disc 0, point 4, speed 19.75 m/s. The final fixed checks at 2 s
and 1.25/0.625 microseconds agree closely on a nonterminal off-core state.
This is a failed terminal-classification premise, not a timestep disagreement.
Do not restart the original collector unchanged: it would spend the full
ladder on the same known nonterminal node and stop again.

The human confirmed that central-core loading is primary and off-center
trapping should be separate. This does not authorize calling retained atoms
escaped or weakening the core rule. The full 6D stability probe for one
off-core equilibrium shows effectively neutral transverse position modes;
do not claim validated 3D off-center trapping from axial confinement alone.
See `validation/offcore_equilibrium_probe.json` in the original output.

Diagnostic continuation: `scripts/complete_mot_strange_diagnostic_masks.py`.
Command: `uv run python -u scripts/complete_mot_strange_diagnostic_masks.py --workers 16`.
Its active sibling output root appends `_diagnostic_completion_v3` to the original
campaign directory. It reuses exact completed rays and all retained nodes,
copies the same pilot geometry, and completes remaining speed nodes while
preserving every -1. New unresolved nodes receive two adaptive checks through
2 s; short fixed edge checks are retained, and unperformed final fixed checks
remain explicitly pending. This pass is diagnostic and cannot certify a final
loading rate. Both collectors share the original OS lock to prevent overlap.

Inspect the diagnostic root's `process.json`, `launcher.json`, `progress.json`,
`latest_capture_bounds.json`, `disc_capture_bounds.npz`, `failure.json` and
`diagnostic_completion.json`. The lower capture integral uses only definitive
central captures; the upper treats indeterminate nodes as possibly captured.
These are bounds of the sampled 0--80 m/s quadrature, with a separately labeled
approximate sampling envelope. Never present their midpoint as an estimated
physical loading rate or conflate the ambiguity width with a 95% statistical
CI. Tail, terminal-classification, retention, precision and aperture gates
remain separate. The primary campaign is still incomplete.

The v2 diagnostic attempt encountered a Numba callback disk-cache
`ReferenceError: underlying object has vanished`. Its outputs and failure
remain under `_diagnostic_completion_v2`. V3 disables disk caching only for
the callback-bearing adaptive helper dispatchers and compiles those helpers
inside each process. Mathematical equations, physical inputs and the original
physical source hash remain unchanged. V3 also reuses the three complete
diagnostic rays from v2. Four capture-bounds/censoring tests and a two-setting
trajectory probe passed before the v3 launch at 20:35:32 UTC. Do not restart
v2; inspect the v3 process and lock before resuming its collector.

V3 completed all 32 x20 pilot masks at 20:37:36 UTC. The unrefined diagnostic
integral bounds are 1,158,543.637 to 1,184,340.661 atoms/s, with a separately
labeled approximate 95% sampling envelope 649,684.633 to 1,702,476.510 atoms/s.
There are 14 indeterminate positive-speed nodes on five rays. No tail-probe
captures or indeterminate tail probes were found. These are sampled-integral
bounds, not a final loading prediction or a scalar estimate plus an error bar.

`scripts/audit_mot_strange_diagnostic_pilot.py` saved the prospective allocation
and indeterminate-node taxonomy. All 46 challenging central-retention cases
passed dual-timestep longer-duration checks. Nine nodes have two adaptive
nonterminal results at 2 s. The other five have delayed adaptive wall exits
but need final fixed validation beyond 400 ms. The companion
`scripts/resolve_mot_strange_pilot_fixed_nodes.py` audits these five nodes at
2 s with 1.25 and 0.625 microsecond steps and saves a separate overlay; it does
not overwrite original masks. Inspect `validation/fixed_overlay_process.json`,
`validation/fixed_overlay_progress.json`, `fixed_overlay.log`,
`fixed_overlay.stderr.log`, `validation/additional_fixed_nodes.json`,
`refined_capture_bounds.json`, and `disc_capture_bounds_refined.npz` before
any repeat. Its launcher is `fixed_overlay_launcher.json` in the v3 root.

The full pilot suggests approximately 70,000--80,000 production rays to attain
a 5% statistical half-width. The conservative nested-variance check suggests
4,000 discs x20 points. This is a prospective estimate, not a frozen allocation
or authorization to reinterpret censored outcomes. The user has been asked
whether to define central capture within a 2 s observation horizon (core rule
unchanged; retained off-core atoms reported separately, never called escaped),
or retain the eventual-capture interpretation with the final rate withheld
while nonterminal states remain. That observation-time question is pending;
do not start dependent larger production until answered.

The five additional fixed-node checks finished at 20:43:30 UTC and all
resolved as delayed wall escapes at both final timesteps. The separate
overlay reduces the diagnostic bounds to 1,158,543.637--1,171,552.416 atoms/s,
with approximate 95% sampling envelope 649,684.633--1,685,486.472 atoms/s.
Nine nonterminal nodes remain. Always use `refined_capture_bounds.json` for
the latest bounds; `latest_capture_bounds.json` preserves the earlier
unrefined result. All original masks and attempted nodes remain unchanged.
The known-capture retention audit passed all 46 cases. Primary production
remains pending the human's observation-time definition.

Independent final-step audit while that answer is pending:
`scripts/audit_mot_strange_remaining_nodes.py`, launched at 20:48:26 UTC.
It checks all nine retained nodes at 2 s and 1.25/0.625 microseconds, reusing
already saved final fixed pairs where present. This is numerical validation
only, and does not reinterpret an off-core retained trajectory as escaped or
captured. It preserves original masks. Inspect the v3 root's
`remaining_fixed_launcher.json`, `remaining_fixed.log`,
`remaining_fixed.stderr.log`, `validation/remaining_fixed_process.json`,
`validation/remaining_fixed_progress.json`, individual files under
`validation/remaining_fixed_nodes/`, and `validation/remaining_fixed_nodes.json`.
Verify actual live PIDs before any restart. Do not start a competing audit or
larger production pool while these workers run.

The remaining-node audit finished at 20:50:55 UTC. All nine final fixed pairs
agree on nonterminal outcomes at 2 s. This completes that numerical check;
it does not resolve eventual capture/escape or change the reported bounds.

The independent full-speed-grid check was launched at 21:03:53 UTC with
`scripts/audit_mot_strange_full_fixed_grid.py` (three workers). It compares
every sampled speed for the two originally predetermined pilot rays
(disc 0/point 0 and disc 17/point 13), plus the challenging captured ray
disc 0/point 4, against fixed RK4 with the retained ladder. Existing validated
fixed pairs are reused, including the nine nonterminal pairs where relevant.
It preserves original masks. Inspect the v3 root's `full_fixed_grid_launcher.json`,
`full_fixed_grid.log`, `full_fixed_grid.stderr.log`,
`validation/full_fixed_grid_process.json`, `full_fixed_grid_progress.json`,
`full_fixed_grid/` and `full_fixed_grid.json`. Check live commands before
any resume. Do not start the larger campaign until this gate passes and the
pending observation-time question has been answered.

The full-speed-grid audit passed at 21:04:42 UTC: all 984 sampled nodes across
the three selected rays matched the dual-fixed-step evidence, with no
classification disagreements. The remaining-nine-node final fixed audit's
maximum endpoint separation was 3.2471159624e-7 m (0.325 micrometers).
These numerical checks are complete; do not restart their pools. The capture
bounds remain unchanged and the larger campaign still awaits the human's
observation-time definition. The pilot does not meet the 5% sampling target,
and aperture/launch-placement convergence is not yet established.

## User-selected finite 2 s definition and velocity refinement

The human has now explicitly selected capture by 2 s with the unchanged
central-core rule, and off-center retention reported separately. This supersedes
the pending-question statements above. There is no remaining observation-time
decision to request. Retention at the horizon is a distinct outcome, never wall
escape or eventual capture, and does not establish off-center 3D trapping.

`scripts/run_mot_strange_finite_capture_campaign.py` converted the exact pilot
evidence under this definition, preserving all original masks and fixed-step
overlays. Its immutable `_core_by2s_v1` output has 640 resolved rays and nine
retained-at-horizon nodes. The provisional central rate is 1,158,543.637 atoms/s,
95% disc-cluster interval [649,684.633, 1,667,402.641], at the unchanged thermal
inputs. The off-core retention integral is 13,008.779 atoms/s, separately from
central capture. This is a finite-observation diagnostic, not a final rate.

V1 stopped before production or allocation freezing: nested 0.5 versus 0.25 m/s
quadrature differs by 26,706.692 atoms/s (2.305%), exceeding the unchanged 2%
gate. Preserve its failure and source snapshot; do not resume it unchanged.

The validated revision is
`scripts/run_mot_strange_finite_capture_refined_campaign.py`, writing the sibling
`_core_by2s_v2_dv0p125`. It reuses only proven coarse-grid nodes, marks incomplete
refined rays as refinement pending, and computes the missing interleaved
0.125 m/s nodes through 80 m/s, retaining the 85--120 m/s tail probes. Ten finite
classification/refinement tests pass. The physical source hash and equations
remain unchanged. Runtime callback disk-cache isolation remains in use.

Command: `uv run python -u scripts/run_mot_strange_finite_capture_refined_campaign.py all --workers 16`.
Inspect its manifest, source snapshot, process/launcher files, logs, progress,
pilot/latest_estimate.json, validation/fresh_finite_workers.json,
validation/full_fixed_grid.json, pilot/retention_audit.json, sampling_plan.json,
production/latest_estimate.json, failure.json and completion.json. The shared
original campaign lock prevents duplicate collectors. Always check live process
commands and parents before resuming, and never modify a live runner.

The revision must pass fresh-worker horizon classification, the refined pilot
2% quadrature gate, fixed checks at every outcome edge, all refined speed nodes
on the same three independent audit rays, and the refined pilot retention audit
before freezing the independent production allocation. The production count is
fixed from pilot variance, with a 1.5 variance safety factor, 20 points per disc,
and a <=5% relative 95% interval target; interim estimates do not control stopping.
All numerical, final retention, statistical precision, aperture/launch-placement,
and final visual-review gates remain required. Runner completion alone is not
overall task completion. Retain the surrogate-field and weak-wing instantaneous
population limitations and all thermal/provenance metadata.

The refined runner was launched at 21:35:22 UTC (launcher uv PID 79236;
actual collector PID 127544, via Python wrappers whose parents were verified).
Fresh spawned-worker finite-outcome checks passed. As of 21:43 UTC the refined
pilot was completing expensive fixed-edge checks; production was not yet
started. Check current files rather than treating these PIDs or counts as live.

`scripts/plot_mot_strange_finite_loading.py <output-root> --mode pilot`
(or `--mode production`) renders saved spectra, pointwise cluster intervals,
separate conservative all-empty-speed bounds, interim loading intervals and
clipped-area distributions. It also writes a plot record that requires actual
visual inspection. The finite-v1 pilot figure was inspected and its numerical
quadrature failure is explicit in the title; that presentation review does not
approve the failed numerical gate or satisfy the final-production visual gate.

## Refined pilot passed; fixed production allocation launched

The refined 32-disc x20-point pilot completed at 21:50:18 UTC. Central-core
capture by 2 s is 1,166,971.096 atoms/s, 95% disc-cluster interval
[658,912.531, 1,675,029.662], relative half-width 43.54%. At the retained
T=296.1792893407517 K and assumed n87=5.45155695704043e12 m^-3,
R/n87=2.140619836761894e-7 m^3/s. These are pilot estimates, not final production.
The separate off-core horizon-retention integral is 10,151.410 atoms/s with
95% half-width 10,865.301 atoms/s; 17 sampled speed nodes remain off-core at
the horizon. This does not establish off-center 3D trapping.

The nested 0.25 versus 0.125 m/s integral difference is 8,427.459 atoms/s
(0.7222%), passing the unchanged 2% gate. All 1,944 full fixed-grid comparisons
on three rays passed at 21:54:38 UTC. All 48 refined pilot central-retention
cases passed at 21:54:51 UTC. The pilot figure was rendered and visually
inspected: labels, separate empty-speed bounds, uncertainty and pending final
acceptance gates are explicit. This presentation review is not the final
production visual gate.

`sampling_plan.json` was frozen at 21:54:51.935 UTC: 3,975 production discs x20
accepted points = 79,500 rays, independent seed 2026102502, variance safety
factor 1.5, target relative 95% half-width <=5%, and no optional stopping.
Production started automatically afterward in the same verified 16-worker pool.
At 22:00 UTC 343 rays were resolved; two complete prefix discs were available,
below the first four-disc reporting checkpoint. No production estimate had yet
been emitted. Always read current progress; these counts are historical.

All 79,500 saved production positions were checked inside the cell. Physical
source, active runner, saved runner snapshot and validation hashes still match.
No worker restart or source change was needed. Final production numerical,
retention, precision, aperture/launch-placement and visual-review gates remain
pending. `monitor_last_report.json` records the latest notified estimate so
unchanged values need not be repeated on later heartbeats.

An independent read-only pilot impact diagnostic found maximum captured impact
13.4967 mm; the three captured rays in the 12--15 mm band contribute only
0.9400 atoms/s to this pilot. This is not a bound on omitted aperture or a
convergence pass. The proposed next geometry checks and correct conditional
sampling/coupling are recorded in `FINITE_LOADING_APERTURE_AUDIT_PLAN_20261025.md`.
No comparison trajectories or extra worker pool have been launched. Their
allocation and acceptance criteria must be frozen before their production,
and their pending status must remain explicit.

The first notified independent production checkpoint is 8 complete discs
(160 rays), saved at 22:04:05 UTC: central loading 476,845.706 atoms/s,
95% interval [99,963.927, 853,727.486], R/n87=8.746963669916597e-8 m^3/s
at the unchanged assumed thermal inputs. The separate off-core retention
integral is 2,393.091 atoms/s with 95% half-width 5,658.761 atoms/s. This early
production subset must not be confused with or combined ad hoc with the
32-disc pilot. Its 79% relative half-width is not the final precision result;
the full fixed 3,975-disc production remains in progress.

## Periodic estimates between collector checkpoints

The collector saves statistics at predefined reporting checkpoints. A slow ray
can hold the next checkpoint while additional complete prefix discs already
exist. For requested periodic updates, the read-only companion
`scripts/summarize_mot_strange_completed_discs.py <output-root> --check-published`
reduces all currently complete prefix discs, without changing any collector
checkpoint, cache, latest-estimate alias, geometry, mask or sampling allocation.
It writes only separate JSON and per-disc velocity spectra/rates under
`monitor_estimates`. The first use reduced 82 complete production discs at
23:01:37 UTC; an independent reduction at 64 discs matched five published
central/off-core rate, uncertainty and quadrature fields to numerical precision.
Validation and an exact reducer source snapshot are preserved there.

Compare disc counts as well as timestamps when deciding whether to notify.
Use the most advanced complete-disc estimate from the collector or monitor;
never regress from a monitor estimate to an older collector checkpoint.
`monitor_last_report.json` records which estimate was actually notified.
These remain descriptive interim estimates and never control stopping. The
fixed 3,975-disc production count and all final scientific gates are unchanged.

## Production failure and failed retention counterexample, October 26

At 04:18:51 UTC production stopped scheduling new rays because disc 773,
point 6, speed 25.125 m/s failed the final dual-fixed classification gate.
`unresolved_ray.json` records the failure. The original ray contains all 648
speed nodes, with this node explicitly -1. The 1.25 microsecond fixed result
met the early core rule, whereas 0.625 microseconds escaped. Both adaptive
settings escaped. Preserve these disagreements; do not restart v2 unchanged.

At 04:31 UTC a read-only py-spy stack inspection of verified collector PID
127544 showed the main thread in ProcessPoolExecutor shutdown, waiting for
already submitted work after the exception. Thus `failure.json` is delayed
until the context manager finishes draining; absence of that file is not
proof of healthy production. Some queued workers continue saving trajectories.
Do not duplicate this pool or stop unrelated processes. Inspect actual live
commands and parents before any subsequent launch. Also inspect
`unresolved_ray.json` on every future monitoring pass.

`scripts/audit_mot_strange_production_failed_node.py` performed a separate
single-process diagnostic without editing the collector or production rays.
All three additional fixed steps, 0.3125, 0.15625 and 0.078125 microseconds,
met the unchanged core rule at approximately 10.522 ms. Exact source,
configuration hashes, failed-ray copy and results are saved under
`validation/production_failed_node_20261026`. This diagnostic alone does not
authorize replacing the failed node or accepting the campaign.

The independent longer check
`scripts/audit_mot_strange_failed_node_retention.py` continued beyond the core
event with early exit disabled. Both 0.15625 and 0.078125 microsecond steps
subsequently reached the glass wall at 44.3978 and 44.3968 ms, respectively.
The atom therefore satisfies the cheap core-event rule but fails central
retention. `retention/completion.json` explicitly records `passed: false`.
This is a converged counterexample to interpreting every early core event as
retained loading, despite the 48 selected pilot retention cases passing.
Do not hide this failure, call it stable trapping, discard the launch, or
report a final physical loading rate from the existing event masks.

A new scientific definition decision is required: retain the existing
core-event-by-2s quantity with transient events/later losses explicitly
separate, or require the core rule plus central retention throughout the
final 25 ms of the 2 s observation. The latter is recommended for central
loading. The previously settled primary central-core output and 2 s horizon
are unchanged; this new question concerns the newly demonstrated loss after
a qualifying event. Do not restart dependent production before the answer
and a documented, validated revision. Preserve all outputs, the fixed
sampling plan, thermal inputs and source provenance. The latest notified
689-disc event-rate estimate remains provisional and is not retention
validated. Numerical, retention, precision, aperture/placement and final
visual gates remain outstanding.

The queued work finished and the collector emitted `failure.json` at
04:35:45 UTC. At 04:44 UTC none of the verified launcher, wrapper, collector
or sixteen worker PIDs remained live. A read-only inventory at 04:45 UTC
found 15,499 saved rays: 15,498 resolved and the one preserved indeterminate
ray. There are 773 complete prefix discs (15,460 rays); the remaining saved
rays are not a complete independent-disc sample. The per-ray status and
SHA-256 inventory is `validation/post_failure_inventory.npz`, with a summary
and digest in the adjacent JSON. Original ray files, collector outputs and
source/snapshot hashes are unchanged. No replacement pool was launched.
These additional core-event masks do not discharge the failed retention
gate; no new retained-loading estimate is being claimed. The new retention
definition question remains pending; stay quiet on unchanged heartbeats.

## Human-approved final central retention revision, October 26

The human selected **Require final central retention** in response to the
transient-core counterexample. No definition question remains pending.
Primary acceptance now requires the unchanged core rule by 2 s AND continuous
residence inside the central 2 mm core throughout [1.975, 2.000] s. A recorded
early core event alone is insufficient. Off-core endpoint retention, other
inside-core endpoints that fail the window rule, observed transient core
events followed by wall loss, and numerical failures remain separate.

The new runner is `scripts/run_mot_strange_retained_capture_campaign.py`,
with helper `scripts/mot_strange_retained_capture.py`. Its distinct output is
the sibling ending `_retained2s_v3_dv0p125`. The physical force source and
configuration remain unchanged. Every new trajectory runs with core early
exit disabled. Retained outcomes require the actual 2 s endpoint and finite
final-window maximum radius. Independent adaptive settings must agree on the
acceptance class, with retained endpoints and window radii within 10 micrometers.
Disagreements escalate to tighter adaptive settings and full-duration fixed
pairs down to 0.15625/0.078125 microseconds. Every resulting primary-outcome
edge is checked with fixed integration. Numerical unknowns still fail closed.

Only provenance-matched, dual-proven old wall losses or no-core horizon
outcomes may be reused. An old early core exit is always recomputed from the
original launch, with the full observation. The 32-disc pilot has 974 such
early-core speed nodes. Reused noncentral proofs preserve their original
source-ray hash, methods and evaluations; no fabricated final-window evidence
is inserted. Secondary prior-core-history disagreements between two proven
wall losses have their own outcome, without changing a known wall loss into
retained loading. Full fixed-grid gates compare physical primary outcomes;
secondary wall-history changes are recorded separately and their event-rate
diagnostics are not retained-loading claims.

Five focused tests pass for full-horizon enforcement, final-window/core-rule
acceptance, transient wall loss, dual-setting endpoint convergence and
rejection of early-core exits as reusable retention evidence. Fresh-process
checks include the failed production trajectory, a pilot central reference,
an off-core reference and a fast wall escape. Only after these pass will the
new pilot run. It must pass the unchanged 2% nested velocity-quadrature gate,
all 1,944 full-fixed node comparisons on the three predetermined audit rays,
and a longer 2.2 s dual-refinement retention audit. Then independent production
allocation is recomputed from the new pilot masks using the same variance
safety factor 1.5 and <=5% relative 95% half-width target, and frozen before
production. The old 3,975-disc event-rate allocation remains provenance only;
no optional stopping or pooling of pilot and production is allowed.

All original outputs and failed gates remain preserved. T=296.17928934 K,
n87=5.45155695704e12 m^-3, signed gradients (+10,+10,-20) G/cm, beam powers,
inside-cell full-sphere 15 mm launches, clipped areas, gravity and deterministic
mean force are unchanged. The numerical/statistical runner does not discharge
the remaining aperture/launch-placement and final visual-review gates.

The v3 runner was launched at 21:42:31 UTC (16:42 Chicago), using
`uv run python -u scripts/run_mot_strange_retained_capture_campaign.py all --workers 16`.
Initial verified chain: launcher 35520 -> wrapper 193196 -> collector 199812.
Fresh-worker checks passed before 21:44 UTC, including the transient wall-loss
counterexample, central retention, off-core retention and fast wall escape.
The pilot then started automatically. No production allocation or new retained
loading estimate exists at this initial stage. Check current saved progress;
the first rays finishing do not imply complete independent direction discs.

The new read-only reducer is
`scripts/summarize_mot_strange_retained_discs.py <new-output> --mode pilot|production --check-published`.
It is separate from the incompatible early-event reducer. Two focused tests
pass for excluding transient events from central loading, the direction-disc
cluster interval, and rejecting the legacy early-event schema. It writes only
`monitor_estimates` and must first match a published checkpoint before being
used for notification. Its test certificate is saved in
`validation/retained_reducer_tests.json`; actual checkpoint comparison remains
pending until the collector publishes one. Never alter live collector source
to obtain intermediate estimates or change the frozen production allocation.

At 22:16 UTC the independent retention reducer matched all six published
statistics at the 8-disc checkpoint, then reduced 12 complete pilot discs
(240 rays): central final-retention rate 1,550,991.604 atoms/s, 95% disc-cluster
interval [470,513.679, 2,631,469.530], R/n87=2.845043382022008e-7 m^3/s at the
unchanged assumed thermal inputs. Separate off-core endpoint retention is
12,448.711 atoms/s with 95% half-width 27,399.418. These are descriptive pilot
statistics, not a production precision result. The complete pilot and all
remaining gates must finish before a production allocation is frozen.
The 12-disc figure was actually inspected and its presentation record marked
reviewed; this is not the final production visual gate. Compatible plotting
now accepts `--estimate <monitor-JSON>` with the matching NPZ and labels the
final25ms criterion explicitly. The old early-event reducer remains unsuitable.

The complete retained-definition pilot finished at 22:27:10 UTC: 32 independent
discs x20 points =640 rays. Central final-retention rate is 1,166,971.096 atoms/s,
95% disc-cluster interval [658,912.531, 1,675,029.662], R/n87=
2.140619836761894e-7 m^3/s at the unchanged assumed thermal inputs. Separate
off-core endpoint retention is 10,151.410 atoms/s, 95% half-width 10,865.301.
The nested quadrature difference is 8,427.459 atoms/s (0.7222%), passing 2%.
The independent reducer matched all six published fields again at 32 discs.

A read-only saved-proof audit verified that all 974 central-positive speed
nodes have actual 2 s dual-setting evidence, the core rule and a bounded final
25 ms, and that none reuse an old early-core exit as retention. Latest pairs
are 195 fixed and 779 adaptive; the largest saved final-window radius is
1.362 micrometers. This is saved-evidence integrity, not a replacement for the
remaining convergence gates. The audit is
`validation/pilot_full_observation_proof_audit.json`.
At 22:32 UTC two of three full fixed-grid rays passed (1296 nodes); the challenging
d000p004 audit was still running with 64 saved nodes and no disagreements.
The 2.2 s longer retention and new allocation remain pending. The actual full
32-disc figure was visually reviewed as pilot presentation only; final
production visual acceptance remains pending.


At 01:31 UTC October 27, all 1944 retained-definition full fixed-grid nodes
passed with no primary disagreements or secondary history changes. All 48
selected longer 2.2 s dual-refinement retention cases also passed; their largest
final-window radius was 1.362 micrometers. The runner froze a NEW independent
production plan at 01:28:01 UTC using the retained-definition pilot variance:
3975 discs x20 points =79500 rays, seed2026102502, variance safety1.5, target
relative95% disc-cluster half-width<=5%, no optional stopping. Its numerical
allocation happens to match the historical early-event allocation, but this
is a newly justified retained-definition plan. Production began automatically
under the unchanged verified collector199812 and16 workers. At this stage
report245 rays were saved, with only one complete prefix disc and no published
production interval. No new rate is reported. Production numerical, longer
retention and precision gates, aperture/launch-placement convergence and actual
final production visual review remain pending. Original evidence is preserved.


At 04:12:49 UTC October27 the retained production phase failed closed at
d210p000,33.625m/s. Every saved fixed setting reached2s outside the core
without a core event; the final0.15625/0.078125us endpoints differ about28.545um,
exceeding the unchanged10um tolerance. The raw code remains numerical unknown.
Failure.json and unresolved_ray.json were emitted immediately; verified collector
199812 and16 workers remained live at04:16UTC and may drain queued work. No
production restart, source edit, tolerance relaxation or overlay was made.
A separate single-process adaptive16/32 diagnostic preserves the failed-ray
copy, hashes and settings under validation/production_failed_node_20261027.
It is diagnostic only and does not waive the failed production gate.
The compatible reducer independently matched the published checkpoint and
reported210 complete discs4200rays: central988131.678atoms/s,95%CI
[813901.959,1162361.398],R/n87=1.812567833611251e-7m^3/s at the unchanged
assumed thermal inputs; separate offcore5895.079+/-2951.411atoms/s95%half.
All production/final numerical, longer-retention, precision, geometry and final
visual gates remain required.

The adaptive16/32 diagnostic completed04:18:16UTC with both trajectories
offcore at2s but27.133um endpoint difference: pair remainsunknown under10um.
A separate64/128 adaptive extension started04:19:20UTC in the nested
adaptive64_128 folder, single process102596 (uv123624), preserving its own
source/manifest and failed-ray copy. This remains diagnostic only; no campaign
classification or tolerance changed.

At04:31UTC the64/128 refinement completion was read: pair remainsunknown,
endpoint difference16.435um and final-window difference13.388um, both above
unchanged10um. Both settings reach2s offcore without a core event. Collector
199812 and16 workers still live;5066resolved rays,1numericalunknown and1running
ray saved, with210complete prefix discs. No new rate notification. The next
256/512 adaptive pair was launched in its own diagnostic subfolder, retaining
all preceding failed evidence; no tolerance or campaign output changed.

At04:46UTC launcher35520,wrapper193196,collector199812 and all16 workers
were verified absent. Post-failure inventory preserves5068rays:5067resolved
and1unknown,210complete prefix discs4200rays. Per-ray SHA256/status and aggregate
digest are in validation/post_failure_inventory.json/.npz; outputs untouched.
Adaptive256 completed;512 threw the unchanged adaptive iteration-limit error,
so its planned pair is incomplete, not passed. A separate saved128/256 comparison
has0.109um endpoint and0.089um window differences, diagnostic only. A bounded
additional full2s fixed pair0.0390625/0.01953125us was launched single-process
under production_failed_node_20261027/fixed_0p0390625_0p01953125us, uv55480.
This can support a prospective numerical revision only after proper validation;
production remains failed and no original mask or tolerance was modified.

At05:31UTC the additional0.0390625us full2s case was saved; second step
still running with verified CPU progress. Its saved final-window maximum is
13.387picometers smaller than its actual endpoint radius, violating the1pm
internal consistency check, so this diagnostic outcome remainsunknown. This
is an observed endpoint/window bookkeeping inconsistency, not a physical
classification disagreement or grounds to loosen10um convergence. Evidence
and rawJSON hash are in production_failed_node_20261027/endpoint_window_bookkeeping.json.
The fixed-classifier loop and final endpoint observation need a separately
validated numerical revision; active source and outputs remain unchanged.

At06:16UTC the additional fixed pair was complete (06:11:39UTC): endpoint
difference33.040um, window difference26.894um, pair remainsunknown; diagnostic
process exited. Thus the bookkeeping issue does not explain away endpoint
nonconvergence. A separate scripts/mot_strange_retained_numerics.py observation
loop now observes every propagated endpoint, including a final floating-point
remainder. It calls the unchanged frozen force/RK4 step, without editing src.
Three analytic tests pass: final-window/trace endpoint inclusion, wall at final
endpoint, core rule at horizon without early return. Certificate:
validation/retained_endpoint_observation_tests.json; no campaign acceptance.
A single-process0.625/0.3125us full2s trace was launched with uv91640 under
production_failed_node_20261027/endpoint_trace_v1 to localize sensitivity,
saving actual sample times, r,v, previous axis, field norm and capture/window
observations. No interpolation, production restart, tolerance or mask change.

At06:31UTC the corrected-observation trace pair was complete: exact endpoints
match the original0.625/0.3125us endpoints, still53.010um apart. Both are offcore
and endpoint/window consistency now holds. The observed difference grows from
about0.049um at5ms to53um at2s (actual trace times retained, no interpolation).
A read-only28-sample public-reference force comparison gives maximum relative
difference3.064e-12; no reference errors. Two finite-difference scales give
positive local6D Jacobian growth rates about2.5--3.4/s on later offcore samples,
consistent with local error amplification, not proof of global trapping or a
complete explanation of nonconvergence. Sampled fields are above the axis floor.
The512 refinement cannot reach2s under the original2million-attempt cap:
even its largest permitted0.5ms/512 step needs>=2048000accepted steps. A separate
scripts/mot_strange_retained_budget.py changes only this finite cap parameter;
three tests pass, including complete integrator AST identity after normalizing
that single budget change and an analytic unchanged-solution comparison.
Certificate:validation/retained_iteration_budget_tests.json. A single-process
512/1024 pair with6million attempts was launched under budgeted512_1024,
uv76516, retaining the same2s horizon, equations, error scales and10um tolerance.
This is diagnostic only; all original failures/unknowns remain, production
has not restarted, and no campaign acceptance is certified.

Prospective v4 numerical revision (documented before launch): the512/1024
full2s diagnostic completed06:42:31UTC, both offcore, endpoint11.629nm and
window9.470nm apart. All128/256/512/1024 full2s evidence is preserved; no
originalunknown was relabeled. A new scripts/mot_strange_retained_capture_v4.py
and run_mot_strange_retained_capture_v4_campaign.py use separate output
_retained2s_v4_dv0p125, unchanged force/configuration/acceptance/10um tolerance,
and the corrected fixed final-observation loop. After all fixed pairs fail,
the generalized extended rule requires a single definitive fixed category and
mutual agreement of the last THREE128/256/512/1024 adaptive results in category,
endpoint and window within10um; every result/evaluation is saved (method2).
No coincidence of one pair or conflicting fixed category can pass. The finite
attempt budget scales with the stricter core-region step cap and remains<=50M;
error scales/equations unchanged. Eleven focused tests pass, including complete
integrator AST equality except budget, analytic observations, three-way span,
category contradiction and early-return rejection. Existing resolved v3 rays
or partial seeds may be copied only into the new root after exact geometry,
horizon/window and inventory hashes validate; per-ray lineage SHA is saved.
All v3 outputs, failure, unknown and inventories remain immutable. Fresh probes
include the actual d210 failure with preserved failed fixed evidence and NEW
high-refinement integrations; no diagnostic overlay sets its outcome. Before
production, v4 must rerun pilot statistics, all1944 full-grid comparisons and48
longer2.2s retention checks. Independent pilot allocation policy is unchanged,
never controlled by production interim rates. All final precision/retention/
geometry/visual gates remain required.

The v4 all-stage runner was launched with uv187820, wrapper128828,actual
Python111800. Shared campaign lock used; fresh-worker validation is first,
including d210p000. No v4 production has started at this launch report.
New root monitor_last_report.json starts at0notified pilot discs and retains
the last v3 report separately; do not combine revisions/phases.

At07:16UTC all v4 processes were verified exited. All FIVE fresh probes passed,
including d210p000 with NEW high-refinement consensus. V4 then failed07:04:57UTC
before pilot masks: the ray-copy path did not create its destination directory
before shutil.copy2, raising WinError3. Failure and raw fresh-probe certificate
are preserved, no production started. A separate v4r1 helper/runner/output
creates the missing parent directory before the exact copy. Its regression
test exercises a nonexistent destination directory and validates byte-identical
copy plus sourceSHA lineage. All12 numerical/observation/budget/reuse tests pass.
The numerical rule, tolerances, equations and physical source are unchanged.
V4r1 will rerun fresh probes, pilot full-grid and longer-retention gates; it
does not reuse a failed infrastructure gate as a pass.

At07:31UTC v4r1 fresh FIVE probes passed and all640 pilot rays were copied
byte-identically from immutable v3 with640 sourceSHA lineage files checked.
The independent retained reducer matched all six published fields at32discs;
monitor_estimates/pilot_discs_00032.json/.npz preserves spectra and statistics.
This is reused pilot evidence, not new independent sampling. Quadrature0.7222%
passes. New full-fixed audits d0p0 and d17p13 passed1296nodes; d0p4 is running
with64savednodes and no disagreements. Longer2.2s, allocation and production
remain pending. All16 workers/collector121472 and unchanged source/snapshot
hashes were verified; no failure or unresolved ray exists in v4r1.

At10:46UTC v4r1 ALL1944 full-fixed comparisons and48 longer2.2s pilot
retention cases passed. New plan froze10:36:57UTC at3975discs x20=79500rays,
seed2026102502, variance safety1.5, target5% relative95%half, no optional stopping.
Production started automatically. The independent reducer matched six published
250disc fields and reduced the highest complete prefix254discs/5080rays:
R1023112.610atoms/s,95%CI[859044.928,1187180.292],R/n87=1.8767346977532208e-7;
offcore5696.831+/-2632.545atoms/s95%half. AssumedT296.17928934K,
n87=5.45155695704e12m^-3. monitor_estimates/production_discs_00254.json/.npz.
All5068 reused/seeded records have source lineage. Actual d210p000v33.625
is offcore under NEW v4r1 proof: last3 highadaptive endpoints span11.629nm,
window9.470nm. Original v3 rawunknown and f88c9905... raySHA unchanged.
All16workers/collector121472 remain live, source/snapshots unchanged, nofailure.
Final production numerical/longer-retention/precision, aperture/placement and
actual final production figure review remain pending; no overall completion.

At 2026-10-28 06:43:52 UTC the user explicitly reduced the horizon for the
remaining work to 1 second and imposed a seven-hour wall-clock cutoff. This
supersedes the prospective no-time-cutoff production policy for the continuation.
The hard deadline is 2026-10-28 13:43:52 UTC (08:43:52 America/Chicago).
The v4r1 pool was stopped by verified process identity after preserving atomic
checkpoints; its source, raw results and original unknowns were not altered.
The stopped inventory records 1,651 complete independent discs, 33,417 resolved
ray files and 33,432 total ray files. In-flight settings remain incomplete.
The new root is `_retained1s_v5_deadline_20261028`, using the remaining original
production disc IDs [1651,3975), with a separate 1-second estimator. Existing
2-second classifications are not relabeled as 1-second evidence or pooled.
Acceptance is the same core event by 1 second plus continuous central 2 mm
residence throughout [0.975,1.000] seconds; the final window remains 25 ms and
the endpoint/window numerical tolerance remains 10 micrometers. Equations,
force inputs, fixed-step pairs and adaptive refinement algorithms are unchanged.
Fresh 1-second probes and full-grid/longer diagnostics get new certificates;
no 2-second certificate is copied into a 1-second pass. Fifteen focused
numerical/observation/cancellation tests passed before launch.
The continuation checks an administrative stop between numerical settings,
starts draining five minutes before the hard deadline, and has an independent
identity-checked deadline guard. Only complete discs enter descriptive nominal
95% intervals; time truncation can select by computational cost, so unconditional
coverage and the original 5% precision target are not certified. Deadline
termination is an incomplete-study report, never a scientific-completion pass.
