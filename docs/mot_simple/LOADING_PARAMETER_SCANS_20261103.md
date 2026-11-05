# Independent conventional two-level MOT loading scans

The user-approved November 3, 2026 campaign uses 780.241209686 nm cooling light,
10 mW per traveling beam, -15 MHz detuning, 12.7 mm Gaussian 1/e² intensity
diameter, and 20 G/cm axial anti-Helmholtz gradient as its reference. Each scan
varies only its named parameter; the diameter scan holds power fixed.

| Scan | Values |
| --- | --- |
| Power | 2 through 40 mW per beam, every 2 mW |
| Detuning | -5 through -40 MHz, every -2.5 MHz |
| Diameter | 5 through 30 mm, every 2.5 mm, plus the 12.7 mm reference |
| Axial gradient | 2 through 40 G/cm, every 2 G/cm |

There are 64 unique physical configurations. Shared reference configurations
are simulated once. Power is displayed as raw and effective single-beam
Gaussian-center saturation, giving five plots from four independent sweeps.
These are not optimization-dependent scans or a Cartesian product.

The vapor is Maxwellian Rb-87 at 296.483 K and Rb-87 partial pressure 1e-10 Torr.
Temperature reweighting holds pressure fixed and updates number density as
`p/(k_B*T)`. It changes the loading integral, not the deterministic capture
trajectories. There is no recoil diffusion, repumper, or atomic-temperature
claim in this effective two-level model.

## Reproduce or resume

From the checkout root:

```powershell
uv sync --all-extras
uv run python scripts/validate_two_level_parameter_campaign.py
uv run python -m pmot.mot_simple.parameter_campaign all --workers 20
```

The independent pilot is excluded from production estimates. It chooses a
fixed number of direction discs for each configuration before production
starts, targeting an 8% relative confidence half-width to allow margin against
the user's preferred 10%. The budget estimate reserves 30% for difficult
audits. The saved plan reports when the budget requires reducing that target.
Observed precision is always reported; a zero sampled mean is not evidence
for an exact zero cross section or a useful relative uncertainty.

All production configurations use 50 uniform-area points per disc, full-sphere
directions, 15 mm disc radius, and 15 mm disc-center distance. All launch
velocities on a disc are parallel. Each configuration uses a prefix of one
saved master geometry, with seed 2026110302. The eventual multilevel campaign
must reuse the exact geometry and sample size for each matched configuration.
The pilot uses a separate seed, 2026110301, and 12 discs by 50 points.

The initial pilot found capture boundaries above 30 m/s. Consequently this
campaign saves 0--80 m/s cross-section spectra at 0.25 m/s spacing and rejects
spectra whose capture evidence approaches the high-speed endpoint. Every
zero threshold receives a direct velocity scan; nonmonotone masks and the
existing fail-closed dual-timestep/duration audit ladder remain authoritative.
Production uses 5 µs RK4 and the exact anti-Helmholtz field. An optional Numba
implementation is checked against the reference field, force, and trajectory
implementations without fast-math, field interpolation, or force interpolation.
The capture rule remains 5 ms continuous residence within the 2 mm core OR
two entries separated by an exit. Independent representative 100 ms diagnostics
check subsequent confinement and endpoint timestep convergence.

Resume or refresh with:

```powershell
uv run python -m pmot.mot_simple.parameter_campaign production --workers 20
uv run python -m pmot.mot_simple.parameter_campaign report
uv run python -m pmot.mot_simple.parameter_campaign audit
```

Source/input signatures prevent incompatible checkpoint reuse. Checkpoints
are written to local application storage every 5,000 rays (the initial resumed
reference point retains its 500-ray interval). Completed points are published
to the repository once, with lossless gzip compression of the full audit CSV.
Bounded retries handle brief Windows sync-client locks on published files.
A failed or budget-limited run retains its evidence and is labeled
incomplete. The deadline is saved in the production plan and survives restarts.

## Products and temperature reweighting

Campaign outputs are under
`outputs/statistics/mot_simple/loading_scans_exact_D2_20261103` and the matching
`outputs/figures/mot_simple` directory. They include:

- the frozen configuration manifest, production plan, independent pilot
  results, validation report, master geometry, and execution status;
- per-configuration capture thresholds, direct velocity masks, endpoint audit
  ledgers, mean cross-section CSVs, and per-disc capture counts with area and
  sample-size metadata;
- per-disc loading integrals, aggregate loading means and 95% Student-t
  intervals, and the final consistency audit;
- five PNG/SVG figures and a self-contained browser HTML report with a vapor
  temperature control that recomputes each disc integral and its Student-t
  interval at fixed pressure;
- `notebooks/mot_simple/loading_scans_exact_D2_20261103.ipynb`, containing saved
  previews and editable plotting/reweighting cells. Run it with the checkout's
  environment, for example `uv run python -m jupyterlab`.

For an independently reweighted interval, integrate every disc's spectrum
first, then calculate the mean and `t_(0.975,N-1) * sample_std/sqrt(N)` across
discs. Do not integrate pointwise cross-section confidence bounds. Retained
null capture counts occur only at authenticated zero-speed nodes; use the
exact weighted-integrand anchor `g(0)=0` without imputing `sigma_capture(0)`.

Preflight timing probes with the superseded velocity domain or interrupted
filesystem writes are retained only as diagnostic provenance and are excluded
from production. The old historical geometry-checksum regression is unrelated
to this new saved geometry; the unmodified committed geometry functions also
produce its observed mismatching checksum on this environment.

No multilevel curve is reused or overlaid in this campaign. That later
comparison requires matched, validated output from the replacement physical
population-rate kernel.

## Completed results (November 4, 2026)

All 64 configurations completed, comprising 1,377,650 launch rays. Each point
uses 251--901 independent direction discs and 50 points per disc. The final
campaign audit and independent verification passed. Published evidence and
geometry checksums, per-disc integrals, mean cross sections, fixed parameters,
and pointwise Student-t intervals were checked. The full browser dataset also
reproduced independent Python reweighting at 100, 296.483, and 500 K with fixed
Rb-87 pressure. All five actual final figures and the final browser report
were visually reviewed; the notebook schema and its five saved figure outputs
were verified.

At the reference settings the loading mean is 17.2884 million atoms/s, with a
pointwise 95% interval [16.6798, 17.8970] million atoms/s (3.52% relative
half-width). The largest sampled means in the independent sweeps are:

| Varied parameter | Setting | Mean ± pointwise 95% half-width [million atoms/s] |
| --- | --- | --- |
| Power | 40 mW/beam (s0 = 37.8387, s_eff = 1.48579) | 25.5940 ± 1.6347 |
| Detuning | -20 MHz | 21.7274 ± 0.8150 |
| Diameter | 27.5 mm | 29.2526 ± 1.7786 |
| Axial gradient | 12 G/cm | 22.2982 ± 0.8042 |

Power continues to increase with diminishing gains over the sampled range.
Diameter has a broad plateau near 25--30 mm, and gradient near 10--12 G/cm.
These are sampled means with overlapping pointwise intervals, not certified
continuous or joint optima. Each other parameter stays at its reference value.

60 of 64 configurations have relative confidence half-width below 10%.
Exceptions are the 5 mm diameter (11.14%) and detunings -35 MHz (10.75%),
-37.5 MHz (12.84%), and -40 MHz (15.16%). No optional stopping or selective
sample pooling was used to hide these wider intervals.

The completed products retain the specified finite 15 mm launch aperture and
launch plane. Monte Carlo intervals do not include model error or residual
numerical error. The full capture-audit evidence includes a documented repair
of timeout minimum-radius bookkeeping: force/RK4 dynamics and capture
decisions were unchanged, the original records were preserved, and the two
affected zero-speed interim records were replayed before resuming. See
`terminal_radius_bookkeeping_migration.json`. The user subsequently authorized
continuation until completion, superseding administrative deadline stopping;
see `continue_until_complete_override.json`. Completion remained within the
original 24-hour target.

Machine-readable final summaries and review records are
`final_results_summary.json`, `independent_final_verification.json`,
`final_audit.json`, and `final_artifact_review.json` in the statistics root.
