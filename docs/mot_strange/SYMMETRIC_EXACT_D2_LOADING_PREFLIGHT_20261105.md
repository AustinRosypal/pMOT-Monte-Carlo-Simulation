# Symmetric strange-MOT loading preflight

This is a conventional 24-state MOT driven by a calibrated optical surrogate
field. It uses the Section 12 population-rate solver for the 780 nm force,
gravity in the `−z` direction, and a terminal cell wall. It omits recoil,
conservative trapping-light forces, and trapping-light scattering.

## Scan settings

- Trapping wavelength 1529.268881 nm, symmetric waist positions at ±10 mm,
  ideal 35 mm input diameter and 80.3 mm lens focal length. The waist radius
  is 2.23363 µm. Each return has the same power as its incident component.
- Cooling wavelength 780.241209686 nm, 27 mW per traveling beam. Reference
  cooling diameter is 12.7 mm at Gaussian 1/e² and reference detuning
  is −15 MHz. Each repump component has 0.1 mW and a 12.7 mm diameter.
  Repump detuning is zero at the ARC F=1 → F′=2 reference. The repump
  wavevector uses 780.232684 nm.
- Both trapping scans use central gradient signs (+,+,−). In the equal-power
  scan the traveling components have powers (P,P,P) on the three axes.
  In the doubled-axial scan they have (P/2,P/2,P). P is 1, 3, …, 39 mW
  plus the 10 mW reference and the 40 mW endpoint. The detuning and
  cooling-diameter scans fix the doubled-axial setting at P=10 mW.
- The trapping plots will use the center intensity of one axial traveling
  component and its signed local field gradient. Record the transverse
  intensities and gradients alongside them. The cooling-beam saturation
  coordinates use the benchmark reference intensity 16.69 W/m² and
  linewidth 6.065 MHz. Report them separately as `s0` and `s_eff`.

The local reference check passed for the 69 selected settings and 22
alternate-sign diagnostic settings. It compared optical intensity, field
signs, force, and populations with the independent beam and full
24-state solver paths. The doubled-axial 10 mW point gives
(+2.6113,+2.6113,−5.2226) G/cm at the origin.
The gravity-on core force has a locally stable axial equilibrium in each
of the 69 selected settings at both Jacobian resolutions. The 10 mW
doubled-axial equilibrium is 5.216 µm below the origin.

Run these checks from the repository root with
`uv run python scripts/validate_mot_strange_loading_scans.py --local-only`.

## Retention and loading

The benchmark core event uses a continuous 5 ms inside a 2 mm sphere or
two entries separated by an exit. The candidate retained-core definition
adds residence inside that sphere throughout the final 100 ms of a 2 s
observation. The off-core screen measures a final 100 ms window outside
the core and within 1 mm of the window's initial position. It is followed
by a separate off-core equilibrium and return test. That test requires a
nearby force zero with gravity included, negative spectral abscissa of
the six-dimensional Jacobian at two resolutions, and return after
independent displacements on all three axes. This identifies verified
equilibrium pockets. Other localized observations remain separate,
including trajectories for which a nearby equilibrium cannot be verified.

At 1 mW one low-speed trajectory passed the localization screen at 1.6 s
and reached the wall by 2 s. A different 1 mW trajectory passed the 2 s
screen while drifting toward the wall. These diagnostics motivated the
two-second check and the separate restoring test. Keep wall escapes,
retained core, verified off-core, other retained, and unresolved distinct.
The original trajectory limits and evidence from preflight remain available
under `outputs/diagnostics/mot_strange/symmetric_exact_D2_loading_20261105`.

The planned geometry samples directions uniformly over the full sphere.
It samples a 15 mm-radius disc at a 10 mm plane distance, clipped to the
strange-MOT cube, and uses the independently calculated area of each clipped
disc. The velocities on each disc are parallel. Rates are aperture-specific.
Use fixed Rb-87 partial pressure of 10⁻¹⁰ Torr and vapor temperature
296.483 K. Use the direction disc as the cluster for a pointwise 95%
Student-t interval. Select and freeze production counts from excluded pilot
directions. A zero observed category has undefined relative half-width.

Production remains gated by numerical closure, velocity-grid convergence
and tail checks, and timing on independently sampled directions. Runtime
must include slow retained trajectories as well as fast wall escapes.
The 0.2 s diagnostic agreed across adaptive and fixed-step classifications
on eight rays but left 103 and 69 speed nodes indeterminate on two of them.
The 1.6 s timing sample included completed full-grid rays between about
5.5 and 191 worker-seconds and rays that exhausted the iteration budget.
Neither diagnostic estimates production throughput on its own.
The early concurrent ARC-initialization probes are excluded from the
timing and classification evidence. One logged a missing ARC SQLite table.
In a same-horizon replay, eight 40 mW equal-power nodes previously marked
localized reached the wall with both adaptive refinements and fixed RK4
at 1.6 and 2.0 s. This discrepancy establishes a reproducibility problem
in the early evidence. The exact runtime cause remains open.

Prepare the ARC-backed force arrays in the parent before spawning workers
and use private worker Numba caches. A fresh-worker check reproduced the
two-second classification on two selected nodes, including the delayed
1 mW core capture. The delayed 15 m/s node met the core criterion after
approximately 0.755 s and passed retention at 2 s with both 5 and 2.5 µs
RK4 steps. The fixed pair took about 39 worker-seconds for that node.
Seven clean two-second full-grid rays finished with
no unresolved positive-speed nodes in 5.69–11.75 worker-seconds. These
seven rays reached the wall at every sampled speed.
The eighth ray, at 1 mW equal power, finished in 416.01 worker-seconds.
It had 41 localized speed nodes and used 28 fixed-step evaluations.
These localized nodes still require restoring-pocket verification.

## Comparison with the regular multilevel campaign

On the same 256 position/velocity states and identical cooling, repump,
gravity and atomic inputs, the median compiled acceleration evaluation
took 7.77 µs with the coil field and 7.26 µs with the surrogate field.
This used five alternating timing batches of 40,000 evaluations per
field after compilation. On the same delayed 1 mW surrogate launch and
the same 5 µs RK4 step, ending at the core event took 4.53 s of compute.
Continuing through the 2 s retention horizon took 12.00 s.

The regular campaign used its audited scalar boundary on 386,765 of
516,480 production rays. Those scalar rays averaged 13.35 adaptive or
fixed integrations each. The strange-MOT timing probe uses 658 adaptive
integrations per ray before any fixed audits or restoring-pocket checks.
The regular campaign checked longer retention on representative
captures, while the strange-MOT probe applies its retention definition
at each sampled velocity.

The eight timing rays are a diagnostic sample with one ray per selected
setting. Their mean is not a production runtime estimate. To forecast
the 176,640-ray pilot-plus-minimum-production design, measure a
representative sample of directions and launch points across settings
with the intended search and retention rules. A scalar shortcut
requires full-mask validation for each capture category it represents.

## Active pilot and figure requirements

The independent pilot started November 6, 2026 at 9:33 a.m. Central.
It uses seed 2026110601, 32 discs by 16 launches per setting, and 20
point-bound workers. Run it from the repository root with
`uv run python -u scripts/run_mot_strange_loading_pilot.py --workers 20`.
The launch record, manifest and progress are under
`outputs/statistics/mot_strange/symmetric_exact_D2_loading_20261105`.
Per-ray checkpoints are kept in the corresponding `LOCALAPPDATA`
campaign directory. Spot replays use
`uv run python scripts/replay_mot_strange_loading_pilot.py 0 --count 1`.
The pilot is excluded from production confidence intervals.

Each final figure must carry its fixed physical and numerical controls,
the exact scan grid and both retention categories. Its data table
records actual per-point powers, intensities, signed gradients, cooling
detuning and diameter, cooling saturation, production sample counts
and intervals. The two power figures have central axial intensity
on the bottom axis and signed axial gradient on the top. The reporter
under `scripts/report_mot_strange_loading_scans.py` takes complete audited
production results and writes the notebook and browser report.

On November 6 the synchronized progress file returned a Windows access
error while the original pilot workers were running. All 512 reference
launches finished and were saved locally. The recovery guard resumed
the same pilot at 10:11 a.m. Central using
`scripts/resume_mot_strange_loading_pilot.py`, which retries publication
and mirrors live progress into the local campaign cache. The original
manifest, checkpoints and error log are preserved. An independent replay
reproduced ten attempts from a reference launch with core retention.

The reference timing sample took 63.02 worker-seconds per ray on average,
with a 5.32-second median. This difference reflects the cost of the slow
trajectories and rules out using the median as a campaign forecast.
`scripts/diagnose_mot_strange_velocity_sampling.py` now compares
probability-weighted sampling of the existing velocity grid against
saved dense masks. This is a timing and precision investigation.
The pilot and its observation criteria continue as launched.

At the completed reference setting, all 36 pilot rays with positive-speed
core retention had a nonmonotone positive-speed mask and failed the
trapped-at-zero premise. A conventional scalar capture-speed search
cannot represent those masks. The velocity-sampling experiment instead
draws each positive-speed node with probability proportional to its
trapezoidal flux weight within a speed stratum. Multiplying the stratum's
sample mean by its full weight preserves the dense-grid integral in
expectation. A separate sampled-node trajectory replay checks both timing
and classifications against the saved dense results. This method remains
diagnostic while independent coverage and convergence checks are pending.

The reference replay matched the dense classification on all 424
sampled nodes across 13 diagnostic rays. The timed replay took 95.4
worker-seconds versus 3778.2 for the saved dense computations on
those same rays. The test deliberately included slow and retained
cases, so the ratio is specific to this sample. The held-out check
uses independent seed 2026110605 for launch directions and seed
2026110606 for sampled speeds. The held-out settings cover the
reference and the trapping-power, cooling-detuning and diameter
extremes. The dense pilot is still the reference run.

The 14 independent held-out rays matched all 452 sampled nodes
across seven settings. Every sampled node in that held-out comparison
reached the wall. Separately, the positive reference ray matched
its coarse classifications at all common nodes after refinement
to 0.125 m/s, and two fixed RK4 steps confirmed core retention
at 2.25 s. The refined integral for that individual ray was
12.5% higher. The prospective production design therefore samples
the finer grid using the same number of speed draws per launch.
It uses the separate 0.25 m/s planning pilot for allocation only.
The completed reference and low-power numerical checks and the
independent planning pilot are required before freezing production.

The 1 mW equal-power retained witness also agreed at all common
nodes on the 0.125 m/s grid and remained in the core at 2.25 s
under both fixed steps. Its individual core quadrature decreased
by 25.7% with refinement. These individual changes are numerical
diagnostics, separate from the direction-disc sampling interval.

At the 20 mm cooling diameter, an unconstrained off-core equilibrium
search probed positions more than 0.25 m from the candidate and outside
the simulated cell. The atomic solver correctly failed closed there.
Independent replay reproduced this at two launch points. The resumed
sampled pilot uses a bounded fallback only for that diagnosed out-of-cell
probe error. The original position, residual, local stability and
return tests still decide whether to verify a pocket. The pilot
checkpoints, original error log and recovery source hashes are retained.

The independent sampled pilot completed all 4,416 planning launches.
Fine-grid replay then matched 29 positive and adjacent speed-node
classifications in the reference and 1 mW equal-power tests. The frozen
fine-grid plan contains 883,200 independent production launches across
all 69 settings. Production started November 6 at 1:35 p.m. Central
with 20 workers. The first 16 checkpoint files passed independent
input, draw and classification checks, and four saved integration
attempts replayed to the stated tolerance. Each completed setting is
queued for independent numerical and statistical audit. The initial
565.38 worker-hour estimate implies 37.7 hours at 75% worker utilization.
Six of 69 planned core intervals are forecast below 10% relative
half-width. The planning pilot verified no off-core captures, so it
does not forecast relative precision for that outcome.

The two unresolved reference speed nodes passed additional adaptive
refinements 16/32 and fixed RK4 steps 78.125/39.0625 ns. They remained
localized outside the core but did not verify a restoring pocket.
The independent audit reads the supplemental reviews by checksum
and preserves the original checkpoints. Two settings finished
checkpointing by 3:39 p.m. Central. Their measured throughput put
the first recalibrated production projection near noon Sunday,
November 8, with substantial variation across settings.
The verified dense diagnostic worker pool was placed at below-normal
Windows priority to give production more compute time. Both jobs
continue running and the projection will be measured again.

The reference production audit passed at 4:23 p.m. Central on
November 6. It checked all 800 direction discs with 16 launch
points each, both supplemental resolutions, and three independent
trajectory replays. The core estimate is 607,065 atoms/s with a
95% Student-t interval of 531,550 to 682,580 atoms/s and a relative
half-width of 12.44%. This setting had zero verified off-core
draws. Localized-only and other retained observations are recorded
separately in the audit and excluded from the verified rate.


## Audited production and report delivery, November 8

Production finished at 03:05 Central on November 8. All 69 settings contain
800 independent full-sphere direction discs with 16 launch positions each,
for 883,200 launch-line checkpoints. Counts remained frozen. Each launch
uses 34 stratified speed draws on the 0.125 m/s grid; repeated draws retain
their statistical multiplicity. Planning and dense diagnostic seeds are
excluded from these production intervals.

All 69 independent point audits passed, followed by the full retained-file
checksum, supplemental-review provenance, geometry and Student-t audit.
An independent check also verified the actual runtime wavelengths, powers,
diameters, helicities, zero-coil and gravity controls, detunings and analytic
central trapping intensities/gradients for all 69 settings. The notebook
schema and regeneration code passed. All four embedded figures and exact
value tables were checked in a real browser; the HTML has no external assets.

Largest sampled core-loading means, in millions of atoms/s with pointwise
95% Student-t intervals, are as follows. These are sampled maxima rather
than continuous optima; many neighboring intervals overlap.

| Sweep | Sampled setting | Core rate [95% CI], million atoms/s |
| --- | --- | --- |
| Equal trapping powers | (1, 1, 1) mW/component | 0.820 [0.736, 0.904] |
| Doubled axial trapping power | (0.5, 0.5, 1) mW/component | 0.893 [0.806, 0.979] |
| Cooling detuning | -17.5 MHz, axial P=10 mW | 0.698 [0.617, 0.780] |
| Cooling diameter | 17.5 mm, axial P=10 mW | 0.757 [0.681, 0.833] |

The axial 10 mW reference is 0.607 [0.532, 0.683] million atoms/s.
The power sweeps fall from their largest sampled mean at the 1 mW endpoint
and then broadly recover. The diameter curve has overlapping intervals
near its 17.5--20 mm maximum. Six of 69 core intervals have relative
half-width below 10%; 63 are wider, with a worst value of 60.85% at
-37.5 MHz. No production interval triggered extra sampling or an early stop.
There are zero observed verified off-core captures, with undefined
Student-t precision. Localized observations without a verified restoring
pocket and other retained observations remain separate audit categories.

These are finite-cell, finite-aperture, finite-velocity-grid, two-second
retention estimates from a deterministic local-steady-state incoherent
24-state surrogate MOT. Sampling intervals exclude model and velocity-grid
errors. They do not establish equilibrium temperatures or full optical
pMOT predictions with conservative Stark forces and trapping-light heating.

The portable self-contained artifacts are
[the notebook](../../notebooks/mot_strange/symmetric_exact_D2_loading_20261105.ipynb)
and [the standalone HTML](../../visualizations/mot_loading/symmetric_exact_D2_loading_20261105.html).
The HTML opens directly in a browser without a Python server. Its figures
and tables are embedded, and its location is outside the ignored outputs tree.

The separate dense diagnostic simulation continues with 20 workers and
independent spot replays. It is excluded from production CIs. Full campaign
completion and removal of the follow-up remain pending that run and its
remaining numerical verification. See `dense_eta_measured.json` for the
measured diagnostic checkpointing projection and its explicit sensitivity
limits; this forecast excludes subsequent numerical reviews.

## User-requested cessation, November 8

The user requested cessation to begin a new simulation. Campaign processes
stopped at approximately 16:13 Central, and the recurring follow-up was deleted.
Console interrupts stopped the independent replay and recovery auditor; the
remaining compiled trajectory processes were terminated after checking their
process identities. Completed atomic checkpoints were retained. In-flight work
and incomplete temporary files are excluded from the saved counts.

All 69 production settings, 883,200 launches, production audits, and verified
notebook/HTML artifacts remain complete and unchanged. The excluded dense
diagnostic stopped with 34,255 of 35,328 checkpoints: 66 complete settings and
463 of 512 lines at the 25 mm cooling diameter. Sixty-five settings had passed
the full per-setting dense evidence audit. The 22.5 mm setting's three numerical
review nodes and the remaining dense verification are unfinished. This cessation
does not certify full completion of the excluded diagnostic campaign.

The cessation record and checkpoint integrity results are in
`outputs/statistics/mot_strange/symmetric_exact_D2_loading_20261105/cessation_record_20261108.json`.
Previously audited checkpoint byte streams were rehashed. Other completed files
passed ZIP CRC and exact input, launch-geometry, and velocity-grid checks.
Automatic simulation and audit continuation are disabled.
