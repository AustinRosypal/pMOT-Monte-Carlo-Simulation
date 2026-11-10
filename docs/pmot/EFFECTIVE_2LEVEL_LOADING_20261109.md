# Effective two-level pMOT loading campaign

This is the deterministic effective Rb-87 `F=2 → F′=3` model. It evaluates
the differential Stark shift from the six focused trapping components and
the radiation pressure from six cooling components at every RK4 stage. The
optical envelope is incoherent, and gravity points along −z. There is no
repumper, applied magnetic field, conservative Stark force, trapping-light
scattering, recoil diffusion, or state-resolved optical pumping. This is
a comparison within the two-level approximation.

## Scan

- Three independent Stark-rank selections: scalar + vector + tensor, scalar
  + vector, and vector only. Keep the same optical inputs for these selections.
- Two exact wavelengths: `1529.268881048958 nm` (single-beam scalar–tensor
  cancellation in the stretched reference) and `1529.273195181438 nm`
  (differential scalar zero). The first condition does not imply a vanishing
  common shift in the six-beam geometry. In the balanced center, the tensor
  terms cancel geometrically, leaving a scalar shift if the scalar rank is on.
- Equal power `(P,P,P)` gives local signed gradients `(+G,+G,−G)`;
  doubled axial `(P/2,P/2,P)` gives `(+G/2,+G/2,−G)`. Sweep axial magnitude
  `G=2,4,…,40 G/cm`: 20 settings per plot and 240 in total. The calibration is
  the analytic origin Jacobian of the differential vector Stark energy divided
  by the Bohr magneton. This is an optical frequency-shift proxy. The
  doubled-axial split matches the local gradient ratio of a conventional MOT.
- Both the incident and returned traveling components on the x, y, z paths
  use propagation-frame helicities `(sigma+,sigma+,sigma−)` for cooling and
  trapping. The trap waists are centered at −10/+10 mm, with 35 mm input
  diameter, 80.3 mm focal length, and equal return powers. Axial power per
  component is 3.8295–76.5905 mW for the cancellation wavelength and
  4.8484–96.9681 mW for the scalar-zero wavelength.
- Cooling: 780.241209686 nm, −17.5 MHz, 27 mW per traveling component,
  12.7 mm Gaussian 1/e² diameter. Nominal single-beam center `s0 = 25.5411`
  and `s_eff = 0.744589`, using `I_sat = 16.69 W/m²` and linewidth 6.065 MHz.
  Those are 780 nm quantities. The 1529 nm model uses power, intensity and
  Stark shifts; it does not define a 1529 nm saturation parameter.

## Sampling and capture

Sample full-sphere incident directions; on each direction disc sample
uniform area and launch parallel to its inward normal. The 15 mm disc at a
10 mm plane distance is clipped to a cube of half-width `sqrt(2) × 10 mm`;
weight each disc by the integrated area of that intersection. The vapor
normalization is 296.483 K and 10⁻¹⁰ Torr Rb-87 partial pressure.

Draw continuous speeds from strata of the incident Maxwell flux, exactly
weighted by `v³ exp(−v²/u²)`. Each stratum's normalization is integrated
analytically. This avoids a monotone capture-speed assumption or a finite
speed-grid quadrature. The largest possible acceleration is bounded by
the photon momentum times `Gamma/2` divided by mass, plus gravity. Above
104.8174 m/s, even maximal opposing acceleration cannot reduce the launch
velocity to the final speed threshold within the full cube diagonal. This
gives a model-specific zero-support bound, while strata below the bound are
sampled directly. The interpolator never extrapolates the polarizability
table.

The core screen requires 5 ms of continuous residence inside a 2 mm sphere
or two entries separated by an exit, then continuous residence in that core
and speeds ≤0.05 m/s during the last 100 ms of a 2 s trajectory. A wall
crossing is terminal. A localized off-core trajectory stays outside 2 mm
with final-window position-box diagonal ≤0.5 mm and speeds ≤0.05 m/s. To
count as verified off-core it must also have a nearby gravity-on force zero,
a stable 6D Jacobian at two difference scales, and return after perturbations
along all three position and velocity axes. Other localized and unresolved
flux is kept separate from both reported capture rates.

Each shot is run to agreement at step caps of 5 and 2.5 µs, or 1.25 and
0.625 µs when the initial pair disagrees. A spatial limiter further shrinks
steps whenever the predictor can cross a narrow trapping envelope. The
preflight covered all controls for signed gradients and origin slopes, and
180 representative trajectories in 36 settings. Those agreed at 0.5 and
2 s and at the tested timestep caps, including deliberately close waist
passes. Separate replay after the pilot and production checks saved shot
classifications and representative longer 3 s retention. Definitive
disagreements require review before reporting a final loading rate.

## Run and outputs

Use the repository root and `uv run python`. The pilot started with 10 workers
on 2026-11-09 at 12:54 p.m. America/Chicago. Its seed `2026110901` and its
8×8 direction-disc sample are excluded from production. From the completed
pilot, freeze one count per setting based on pilot direction-disc variance
and feasible runtime. Production uses independent seed `2026110902`, with
common production directions and speed draws across conditions for paired
wavelength comparisons. Report pointwise 95% Student-t intervals by disc;
relative precision is undefined when no events are observed. Do not stop or
expand individual settings based on the measured production intervals.

```powershell
uv run python scripts/validate_pmot_2level_loading.py --workers 10
uv run python -u scripts/run_pmot_2level_loading.py pilot --workers 10
uv run python -u scripts/audit_pmot_2level_loading.py pilot --workers 10
uv run python scripts/run_pmot_2level_loading.py plan --max-discs 128
uv run python -u scripts/run_pmot_2level_loading.py production --workers 10
uv run python -u scripts/audit_pmot_2level_loading.py production --workers 10
uv run python scripts/report_pmot_2level_loading.py
```

The commands are sequential. Review the pilot audit and use its measured
throughput to choose the production cap before freezing; 128 is the default.
On recovery, verify the old process has ended and rerun the same phase with
the same frozen source and counts. Checkpoints are written atomically; the
runner checks each against independently regenerated parent samples before
reuse. No changes to an older MOT campaign are needed.

The early uniform-area checkpoints missed impact parameters below 1.5 mm
across several independent directions. Use the separate impact-stratified
pilot to sample the small central acceptance and still preserve the original
15 mm aperture and every outer annulus. The edges are 0, 0.2, 0.5, 1, 2, 4,
8 and 15 mm. The pilot samples `2,2,2,2,1,1,1` points per annulus on each
of eight directions. Production doubles those per-annulus counts. Weight
each annulus by its exact cell-clipped area. The two impact-stratified seeds
are `2026110913` (pilot) and `2026110914` (production), both independent of
the original uniform design. Preserve the uniform pilot for provenance; do
not pool it with the new design. Never run the two pools concurrently.

```powershell
uv run python -u scripts/run_pmot_2level_impact_loading.py pilot --workers 10
uv run python -u scripts/run_pmot_2level_impact_loading.py audit pilot --workers 10
uv run python scripts/run_pmot_2level_impact_loading.py plan --max-discs 80
uv run python -u scripts/run_pmot_2level_impact_loading.py production --workers 10
uv run python -u scripts/run_pmot_2level_impact_loading.py audit production --workers 10
uv run python scripts/run_pmot_2level_impact_loading.py report
```

The second design's outputs are isolated under `impact_stratified`. Confirm
the audit and check unresolved flux, retained categories, and the observed
pilot variance and wall time before freezing production. Use a fixed
detection sample when the pilot has too few nonzero discs to estimate a
relative precision target. Capture in a low-speed or narrow-impact region
does not establish a nonzero rate at all other settings.

The impact pilot finished at 14:22 America/Chicago on 2026-11-09. It
completed 1,920 batches with 4,689 retained core trajectories, 82 unresolved
trajectories, two unverified localizations, and zero verified off-core
captures. All 34 audit replays passed. Both localized trajectories remained
outside the core under the longer replay, but their local stability tests
found zero spectral abscissae, so they remain unverified. At the 2 G/cm
doubled-axial settings, a common launch carried most of the unresolved flux.
Its representative 3 s replay crossed the cell wall at 2.11 s. Some other
unresolved replays reached the core by 3 s. Report the fixed 2 s estimand and
keep unresolved flux visible beside the sampling intervals.

Production uses 18,960 frozen direction-disc batches across 240 conditions,
with 22 trajectories per speed stratum per direction disc. The 80-disc cap
was selected for an approximately 24-hour compute budget on 10 workers.
Extrapolation from the pilot uses measured per-setting batch times and doubles
the sample per batch. It projects 24.1 hours at cap 80 and 38.2 hours at cap
128. The same pilot variance model projects a median 31% relative half-width
and no setting below 10% at cap 80. Its eight-direction variance estimates
are noisy. Actual precision and numerical ambiguity will be reported from
the independent production sample without changing the frozen counts.

The handoff process is recorded in `impact_handoff.json` and its logs. It
waits on the verified uniform pilot PID, checks that all 240 uniform pilot
summaries exist, and then starts the 10-worker impact pilot. Do not start a
duplicate pool while that handoff or the impact pilot is active. After
reporting, run `uv run python scripts/verify_pmot_2level_loading_report.py
--impact-stratified` to check the 12 PNGs, embedded HTML and clean-kernel
notebook execution.

Progress, controls, raw checkpoints, phase summaries, and audits are under
`outputs/statistics/pMOT_2Level/loading_20261109`. Final targets are
`notebooks/pMOT_2Level/loading_20261109.ipynb` and
`visualizations/pMOT_2Level/loading_20261109.html`; the twelve PNGs go under
`outputs/figures/pMOT_2Level/loading_20261109`. The report generator requires
independent production and a passing replay audit.
