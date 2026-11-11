# Physical multilevel comparison with the completed two-level benchmark

This is a new campaign using the replacement Section-12 24-state Rb-87
population-rate kernel. No historical saturated-per-transition outputs are
reused. The user authorized the same physical benchmark controls and exact
saved launch coordinates, additional repump studies, and completion even if
runtime exceeds the earlier 24-hour target.

## Confirmed controls and grids

The common reference is 10 mW per cooling traveling beam, -15 MHz cooling
detuning from F=2 to F'=3, 12.7 mm Gaussian 1/e² cooling diameter, 20 G/cm
axial anti-Helmholtz gradient magnitude, gravity in -z, and no recoil diffusion.
The cooling recoil/Doppler wavelength is explicitly **780.241209686 nm**.
The repumper has six Cartesian traveling components, 0.1 mW per component,
12.7 mm 1/e² diameter, and zero detuning relative to F=1 to F'=2.

| Family | Grid | Other controls |
| --- | --- | --- |
| Cooling power | 2--40 mW/beam, every 2 mW | Reference |
| Cooling detuning | -5 to -40 MHz, every -2.5 MHz | Reference |
| Cooling diameter, fixed repump | 5--30 mm, every 2.5 mm, plus 12.7 mm | Repump diameter 12.7 mm; powers fixed |
| Axial gradient | 2--40 G/cm, every 2 G/cm | Reference |
| Cooling diameter, matched repump | Same diameter grid | Repump diameter equals cooling diameter; powers fixed |
| Repump power | 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 mW/beam | Reference cooling and geometry |
| Repump diameter | Same diameter grid | Cooling diameter 12.7 mm; powers fixed |
| Repump detuning | -10 to +10 MHz, every 1 MHz | Reference |

There are 124 unique physical configurations and 3,740,750 planned launch
rays. Raw and effective cooling saturation are two views of the same power
data, so the multilevel-only report has nine plots. The comparison report has
six plots: the original five views and a second cooling-diameter comparison
with matched repump diameter. Repump-only studies have no two-level analogue.

The nominal saturation axes use the identical benchmark single-beam Gaussian
center definition, I_sat = 16.69 W/m² and ordinary-frequency linewidth 6.065
MHz. These quantities label the x axes; the physical population-rate kernel
never inserts a two-level 1+s closure. It retains ARC dipoles and each excited
state's pairwise-summed decay rate (linewidths approximately 6.065786--6.066086
MHz), hyperfine offsets and Landé factors as intrinsic atomic-model changes.
The repump frequency is referenced to the ARC F=1 to F'=2 resonance; its fixed
wavevector reference wavelength remains the multilevel builder's 780.232684
nm convention. Detuning is applied to the atomic transition frequency, with
angular frequencies used consistently inside the multilevel kernel.

The physical restoring propagation-frame helicities are sigma+ on x and y and
sigma- on z, in both propagation directions for cooling and repump. This
realizes the two-level model's effective (+,+,-) axis signs. The scalar
two-level approximation has no state-resolved polarization decomposition;
its decorative beam labels are not its force prescription. The comparison
manifest records this modeling boundary rather than equating those labels.

## Geometry, statistics, and numerical checks

Every common configuration reads its exact coordinates and sample size from
`outputs/statistics/mot_simple/loading_scans_exact_D2_20261103`. No new random
geometry is generated. The saved master contains 901 full-sphere direction
discs, each with 50 uniform-area points, 15 mm disc radius, and 15 mm distance
from the origin to the launch plane. Velocities are parallel to each disc's
inward normal. Common configurations use the same 251--901 disc prefix as the
two-level reference; matched cooling/repump diameter points use the same
prefix as their fixed-repump counterparts. Repump-only studies use all 901
discs. The vapor is Rb-87 at 296.483 K and partial pressure 1e-10 Torr, with
pressure held fixed during later temperature reweighting.

Intervals are pointwise 95% Student-t across independent direction discs,
with N-1 degrees of freedom. The fixed matched production counts are not
selected or stopped based on the multilevel production interval. Below 10%
relative half-width is a target, not a guarantee. Any wider interval or zero
sampled loading is reported honestly. Future precision extensions must be
predeclared independent confirmations or otherwise use justified inference;
do not silently pool CI-triggered samples and claim unchanged coverage.

The same 5 µs fixed RK4 base step and original 2 mm-core capture criterion
(5 ms continuous residence OR two entries separated by an exit) are used.
All force stages invoke the public physical population-rate kernel. Positive
boundaries receive 200 ms dual-step checks; failed scalar premises trigger
complete searches capped at 400 ms and an authoritative direct velocity mask.
Zero thresholds are scanned directly. Unresolved direct-grid nodes use the
bounded 250/400 ms and 1/2 s dual-step ladder. Only a literal finite but
indeterminate zero-speed node may remain unknown; omit sigma(0) and use only
the exact loading-integrand anchor g(0)=0. Positive-speed unknowns fail closed.
The saved spectrum is 0--80 m/s at 0.25 m/s spacing, matching the benchmark;
capture near the upper endpoint requires a dense extension before reporting.

The new preflight compares local forces, populations, net beam rates, and
RK4 stages against the public full 24-state reference for every setting.
Selected repump extremes also undergo frozen-environment Gillespie internal
state/force comparisons. Independent baseline trajectory checks and
per-configuration representative longer fixed-RK4 trajectories check
subsequent confinement and timestep convergence. These diagnostics supplement
the original capture rule; they never relabel outcomes to force acceptance.

The local-steady-state, incoherent-light mean-force approximation remains a
model assumption. Frozen-environment event tests are not a full dynamical
adiabatic-relaxation or recoil validation. Weak-wing population-relaxation
limitations remain explicit. No temperature prediction is made.

## Run and resume

From the repository root:

```powershell
uv sync --all-extras
uv run python -m pytest tests/mot_multilevel --import-mode=importlib -q
uv run python scripts/validate_matched_multilevel_benchmark.py
uv run python -u -m pmot.mot_multilevel.matched_campaign production --workers 20
```

`probe` runs a 24-ray timing diagnostic on saved launch points, excluded from
production estimates. The initial fixed-step probe projects a multi-day to
week-scale campaign on the current CPU; refine that estimate from completed
production settings. It does not authorize reduced samples or changed physics.

All computation is resumable ray by ray. Local application storage under
`LOCALAPPDATA/pmot-campaigns/matched_two_level_exact_D2_20261104` holds live
checkpoints, avoiding Synology locks and repeated replacement of large audit
files. Completed settings are published to the repository, with every ray's
evaluated nodes and authoritative mask retained in a lossless archive.
Source, input, geometry, and library signatures reject incompatible resumes;
an OS-held lock prevents duplicate pools. Do not edit frozen source while a
production pool is running. Preserve originals and document/revalidate any
necessary implementation revision.

Shared benchmark sweeps run first so complete comparison curves become
available before the added repump studies. Check `status.json`,
`launch_process.json`, the exact process tree, timestamped logs, and local
per-point metadata. An old failure file is provenance; compare its timestamp
to the verified current process and logs. Never stop unrelated lab processes.

## Outputs

The campaign name is `matched_two_level_exact_D2_20261104`:

- Statistics: `outputs/statistics/mot_multilevel_population_rate_v1/<name>`.
- Multilevel figures:
  `outputs/figures/mot_multilevel_population_rate_v1/<name>`.
- Comparison figures:
  `outputs/figures/model_comparisons/<campaign-name>`.
- HTML previews: [MOT loading report archive](../../visualizations/mot_loading/archive/2026-11-04_original_multilevel/).
- Notebook: `notebooks/mot_multilevel/<name>.ipynb`.
- Original comparison notebook:
  `notebooks/model_comparisons/loading_2level_mot_vs_multilevel_mot_exact_d2_20261104.ipynb`.
- Refined-sampling comparison notebook:
  `notebooks/model_comparisons/loading_2level_mot_vs_multilevel_mot_exact_d2_refined_20261104.ipynb`.

Both notebooks contain saved figure previews and editable plotting cells.
Both HTML files are self-contained browser visualizers. Temperature changes
reweight each saved disc's spectrum and recompute means and intervals at
fixed pressure without new trajectories. Browser capture counts are packed
losslessly as bytes; 255 means an authenticated indeterminate zero-speed
count and is never treated as a positive-speed capture decision.

The immutable comparison manifest verifies common cooling controls, coil
geometry/current, gravity, coordinate prefixes, sample sizes, velocity grid,
and core criterion against the completed two-level files and their hashes.
Repump light and state-resolved atomic dynamics are explicit model differences.
Only validated completed physical settings can enter overlays. A final audit
checks all expected configurations, publication checksums, numerical gates,
reweighting consistency, and retained capture spectra before final completion.
