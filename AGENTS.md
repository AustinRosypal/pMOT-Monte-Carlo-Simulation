# pMOT Monte Carlo project

## Purpose and scientific roadmap

This repository models laser cooling and trapping of neutral rubidium-87. The
eventual goal is a pseudo magneto-optical trap (pMOT): replace the MOT magnetic
field with spatially varying vector AC Stark shifts produced by trapping light.
The vector Stark shift should act as a state-dependent fictitious magnetic
field, allowing cooling and confinement with optical fields alone.

Work must proceed in validated stages:

1. Finish and rigorously validate the deterministic effective two-level MOT in
   `src/pmot/mot_simple`.
2. Rebuild a multilevel Rb-87 MOT starting from that validated implementation.
3. Validate the new multilevel MOT before introducing pMOT trapping light.
4. Build and validate the no-coil pMOT in two sub-stages: first establish the
   focused trapping-light geometry and intensity fields; then introduce the
   hyperfine-resolved scalar/vector/tensor AC Stark Hamiltonian, forces,
   scattering, and trajectory dynamics. Optimize powers and gradients only
   after those physics layers pass their validation checks.

The former 24-state multilevel implementation is archived in
`src/pmot/mot_error`, with its documentation, notebooks, and regression tests
under the corresponding `mot_error` directories. It used a saturated
two-level scattering-rate expression as the elementary bidirectional rate in
a multilevel population matrix. Because the two-level saturation construction
already embeds a closed two-state population response, that closure is not a
valid physical multilevel population-rate model. The archive is retained only
for provenance and code-comparison purposes. Its forces, trajectories,
capture/loading results, temperatures, and any pMOT dynamics that inherit its
dissipative kernel are scientifically invalid and must not be cited as physical
predictions.

`src/pmot/mot_multilevel` now contains the replacement 24-state
population-rate kernel specified by Section 12 of
`PopulationRateEq_Instructions.md`. ARC precomputes the dipole matrix elements,
transition frequencies, pairwise spontaneous rates, excited-state total decay
rates, and hyperfine Lande factors once. The local solver uses
`W = Gamma_e*|Omega|^2/(Gamma_e^2 + 4*Delta^2)`, never the archived saturated
two-level closure, and calculates force from the beam-resolved net rate
`W_b*(p_g-p_e)`. Its deterministic trajectory driver recomputes the full local
problem at every RK4 stage. Core equation and force-symmetry tests pass, but no
capture/loading or temperature campaign is yet validated for this solver. The
new `diagnostics.py` and `capture.py` modules and the three ipywidgets notebooks
under `notebooks/mot_multilevel` provide trajectory, animation,
capture-velocity, cross-section, and loading-rate scaffolding. Capture
timeouts fail closed; these interfaces do not waive the convergence policy.
The seeded ten-atom Section-12 demonstration and its half-step endpoint audit
live under `scripts/run_ten_atom_population_campaign.py`,
`scripts/audit_ten_atom_population_campaign.py`, and the
`ten_atom_section12_probe_20260919` output roots. Its 5 trapped/5 escaped
result and fixed-vapor loading quadrature are diagnostic only: ten direction
discs with one point each and scalar capture boundaries do not establish a
quantitative cross section or loading rate.
Shared anti-Helmholtz calculations and plots remain in
`src/pmot/magnetic_fields.py` and `src/pmot/magnetic_field_plotting.py`; there
is no generic `mot` package.

## Authoritative two-level MOT assumptions

- Atom: Rb-87 represented as an effective two-level D2 atom.
- Cooling detuning: -15 MHz (ordinary frequency; negative means red detuned).
- Cooling power: 20 mW per beam.
- Six cooling beams: counterpropagating pairs on x, y, and z.
- Beam diameter: 12.7 mm unless the user explicitly changes it.
- Default anti-Helmholtz axial gradient: 10 G/cm.
- Gravity is part of trajectory dynamics and points in -z.
- Scattering, detuning, and linewidth quantities in `mot_simple` use ordinary
  frequency units (Hz), never angular-frequency units.
- Effective detuning is `Delta_0 - k.v/(2*pi) - Delta_B`.
- Keep the current simplified Zeeman prescription:
  `Delta_B = xi * (mu_eff/h) * dot(B, k_hat)`, with the axis-dependent effective
  signs in `mot_simple/configuration.py`.
- Every RK4 stage must recompute magnetic field, local beam intensities,
  saturation, Doppler and Zeeman shifts, scattering rates, and force.
- The two-level model is a deterministic mean-force model. It does not include
  recoil diffusion and must not be used to claim a Doppler-limit temperature.
- The September 2026 two-level comparison campaign lives entirely under
  `mot_simple` and deliberately omits temperature plots.  It reuses the seeded
  full-sphere geometry from the multilevel campaign (seed 20260903): 25
  direction discs by 25 uniform-area points per disc at a 15 mm disc radius.
  Treat the 25 discs as the independent clusters for 95% Student-t intervals
  (24 degrees of freedom).  Its loading grids are the explicitly requested 24
  raw-saturation values from 0.25 through 125, 20 effective-saturation values
  from 0.25 through 5 in steps of 0.25, and detunings from -0.5 through -6 in
  steps of -0.25 linewidth.  Raw/effective saturation scans vary only power at
  -15 MHz; the detuning scan holds 27 mW per cooling beam.  The dense force
  diagnostics use the same detuning limits in 0.05-linewidth steps, with
  gravity excluded from radiation-force calculations.
- Campaign capture audits fail closed.  Every zero scalar threshold is scanned
  directly on the loading velocity grid.  If a positive scalar boundary is
  censored or incompatible in the 200 ms dual-timestep audit, repeat the whole
  boundary search at 250 ms and, only if still needed, 400 ms.  Preserve every
  evaluated node.  Once a positive scalar search has failed its premise, use a
  separately converged 0--30 m/s by 0.25 m/s dual-timestep boolean capture mask
  as that ray's authoritative loading evidence even if the mask is monotone;
  never select or average conflicting scalar thresholds.  Complete scalar
  boundary re-searches remain capped at 400 ms.  Only unresolved direct-grid
  nodes may escalate to a 1.0 s and then a final 2.0 s dual-timestep check at
  1.25 and 0.625 microseconds.  Both steps must agree on a definitive
  trapped/escaped result at every positive speed.  If, after that full ladder,
  only the literal zero-speed node remains finite but non-definitive, retain it
  as explicitly indeterminate rather than relabelling it escaped.  Omit
  `sigma_capture(0)` from the reported cross-section spectrum and evaluate the
  loading quadrature with only the exact weighted-integrand anchor `g(0)=0`;
  never impute a zero-speed cross section.  This preserves the unchanged
  core-residence/two-entry trapped definition and is admissible only because
  the incident flux at exactly zero speed is identically zero.  The 1.0 s
  level resolves the documented delayed-capture ray at detuning -3.75
  linewidths; the 2.0 s level documents the converged-but-nonterminal
  zero-speed ray at detuning -4.25 linewidths.

## Polarization convention

Specify beam polarization as `pi`, `sigma+`, or `sigma-`; avoid ambiguous RCP/LCP
labels. Here sigma+/sigma- are helicities defined from the perspective of the
propagating beam (the observer looks along k). A local atomic sigma+/pi/sigma-
decomposition relative to a magnetic-field quantization axis is a separate
operation required in the current multilevel model.

## Capture-velocity convention

- Sample incident directions uniformly in solid angle over the full sphere by
  default. Restricting launch directions to a symmetry octant requires an
  explicit user instruction. Full-sphere cross sections and loading rates use
  the direction-disc average directly, with no octant or `4*pi` multiplicity
  factor.
- A sampling disc is perpendicular to its incident direction.
- All launch velocities on a disc are parallel to the disc normal; offset
  points do not individually aim at the origin.
- Sample disc points uniformly in area.
- Current early-exit trapped criterion: an atom is trapped if it either remains
  continuously inside the central 2 mm-radius core for at least 5 ms, or enters
  that core twice with an intervening exit. Either route is sufficient. This is
  deliberately inexpensive and must be convergence-checked against longer
  bounded-trajectory diagnostics.
- Capture-speed binary search requires an explicitly trapped lower bound and
  untrapped upper bound and assumes local monotonicity with incident speed.

## Validation policy

Do not proceed to multilevel or pMOT claims until the two-level MOT verifies:

- zero force at the origin for symmetric light with gravity excluded from the
  radiation-pressure force check;
- red-detuned velocity damping;
- restoring force on x, y, and z;
- reversal under polarization-sign or magnetic-gradient reversal;
- field-free and zero-detuning symmetry;
- RK4 timestep convergence;
- the expected radiation-pressure force scale;
- a zero and locally linear anti-Helmholtz field;
- stable capture classifications under smaller timestep and longer timeout.

Keep gravity enabled for physical trajectories, but isolate it when validating
the symmetry of the optical/magnetic force law. Force-curve figures should show
radiation pressure explicitly and state when gravity is excluded.

## Repository map

- `src/pmot/mot_simple`: authoritative current two-level MOT, sampling, plots,
  and loading-rate analysis.
- `src/pmot/mot_error`: archived, physically invalid 24-state implementation,
  including its old rate engine and isolated Gillespie/event layers. Use only
  for provenance and regression against historical artifacts.
- `src/pmot/mot_multilevel`: replacement physical 24-state population-rate
  kernel, ARC atomic-data precomputation, cooling/repump beam construction,
  steady-state force evaluation, and deterministic RK4 trajectories.
- `src/pmot/pmot`: pMOT branch. Its no-coil apparatus, single-frequency focused
  trapping-light geometry, vectorized polarizability interpolation, and an
  explicitly provisional differential-transition AC-Stark detuning layer are
  implemented. The latter is not a unique 24-state Stark Hamiltonian and must
  not be treated as a quantitative pMOT prediction; see
  `docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md` and
  `docs/pmot/PMOT_AC_STARK_THEORY.md`, and
  `docs/pmot/PROVISIONAL_AC_STARK_MODEL.md`.
- `src/pmot/configuration.py`, `beams.py`, `fields.py`: shared apparatus and beam
  geometry.
- `src/pmot/magnetic_fields.py`, `magnetic_field_plotting.py`,
  `launch_geometry.py`, `capture_statistics.py`, `loading.py`, `state.py`, and
  `beam_plotting.py`: model-neutral field, launch, capture-analysis, loading,
  state, and visualization primitives. Model packages normally depend only on
  these shared modules. Existing provisional pMOT code explicitly imports the
  archived `mot_error` local-environment kernel only to keep historical
  diagnostics reproducible. That dependency is invalid for physical pMOT
  dynamics and must eventually be replaced by a validated public entry point
  from the new `mot_multilevel` package. All pMOT-specific environment, Stark,
  geometry, trajectory, and output code must remain under `src/pmot/pmot`.
- `data/raw/pmot`: differential-polarizability datasets for the later pMOT phase.
- `notebooks/mot_simple`: current interactive validation and sampling notebooks.
- `tests`: automated physics and numerical checks.
- `docs/pmot/DIAGNOSTIC_TESTS.md`: canonical ordered pMOT construction-QA
  procedure. Generated campaigns live under `outputs/diagnostics/pmot` and
  must stop at the first failed test, with every later test explicitly marked
  not run.
- `docs/shared/BFIELD.md`, `docs/mot_error/ZEEMAN.md`, and
  `docs/mot_simple/SAMPLINGALGORITHM.md`: historical derivations and
  requirements. Every file under `docs/mot_error` is an archive of the invalid
  former solver and is not a specification for the rebuild. New derivations
  and architecture belong under `docs/mot_multilevel`. The root
  `PopulationRateEq_Instructions.md`, especially Section 12, governs the new
  solver. This file and explicit user decisions take precedence if documents
  conflict.

## Authoritative replacement multilevel MOT assumptions

The detailed August and September 2026 campaign records below describe
historical `mot_error` artifacts. Preserve their provenance, but do not use
their numerical results to validate or characterize the rebuilt MOT.

- Atom: state-resolved Rb-87 D2 system with 8 ground and 16 excited states.
- The rebuilt multilevel MOT defaults to 27 mW per cooling traveling beam,
  -15 MHz cooling detuning, 0.1 mW per repump traveling beam, and 12.7 mm
  Gaussian 1/e^2 beam diameter. The September 19 ten-atom diagnostic is an
  immutable 20 mW/beam historical run; its scripts pin that power explicitly.
- The default multilevel trajectory and capture RK4 timestep is 5 microseconds.
  The first 25-disc by 20-point capture/loading output used 20 microseconds;
  preserve it as a historical comparison and save 5-microsecond results under
  a distinct run name.
- Include cooling light and the repumper; the repumper transition graph must
  retain all relevant F=1 channels, including F=1 -> F'=0.
- Precompute all allowed hyperfine-Zeeman dipole matrix elements with ARC and
  derive pairwise spontaneous rates from them. Sum the pairwise rates to obtain
  the decay constant of each excited state. No ARC call belongs in the local
  force or trajectory hot path.
- Use the Section-12 stimulated coefficient
  `W_ge^(b) = Gamma_e*|Omega_ge^(b)|^2 /
  (Gamma_e^2 + 4*Delta_ge^(b)^2)`. Never add a two-level `1+s` saturation term;
  saturation must emerge from the coupled steady-state populations.
- Assemble stimulated absorption and emission symmetrically, add spontaneous
  decay only from excited to ground states, choose diagonals so every column
  sums to zero, and solve `M p_ss=0` with `sum(p_ss)=1`. Reject nonconserving,
  non-normalized, materially negative, or high-residual solutions.
- Calculate each beam force as
  `hbar*k_b*sum_ge(W_ge^(b)*(p_g-p_e))`; do not use ground-population-only
  absorption as the force.
- Use angular-frequency units consistently inside `mot_multilevel`.
- Recompute local intensities, Doppler and Zeeman shifts, polarization
  decomposition, state populations, scattering, and force at every required
  trajectory evaluation.
- Production capture/loading trajectories use deterministic multilevel mean
  force with recoil diffusion disabled so classifications are reproducible and
  can support the local-monotonicity assumption. Bracket and scan checks must
  still verify that assumption in representative regimes.
- A production temperature trajectory requires a separately derived and
  validated recoil-diffusion model. The new deterministic kernel currently
  implements mean force only.
- For the August 2026 relationship campaign, the completed raw-saturation and
  effective-saturation loading sweeps retain their 30 full-sphere direction
  discs by 30 uniform-area launch points per disc. The restarted detuning
  loading sweep uses 15 full-sphere direction discs by 15 launch points per
  disc, and the corresponding temperature sweep uses 15 independent preloaded
  clouds by 15 atoms per cloud. Treat the 15 direction discs or clouds as the
  independent clusters for Student-t intervals (14 degrees of freedom). Keep
  the 30x30 and 15x15 products in separately named output roots so their sample
  sizes and provenance cannot be confused.
- The September 2026 refinement is a separate resumable campaign under the
  `refined_relationships_full_sphere_25x25_r15mm_27mW_reference_repump0p1mW_20260903`
  output root. Its three loading sweeps are independent and sequential, never a
  Cartesian product; every point uses 25 full-sphere direction discs by 25
  uniform-area points on a 15 mm disc. Its grids are the retained saturation
  values through 50 plus 60, 70, 80, 90, 100, 110, 120, and 125; effective
  saturation 0.25 through 5.00 by 0.25; and loading detuning -0.5 through -6.0
  by -0.25. Saturation studies vary cooling power to realize the requested
  value at -15 MHz and mark 27 mW as the reference; detuning, temperature, and
  force stages fix cooling components at 27 mW and repump components at 0.1 mW.
  Temperature uses 25 independent preloaded Langevin clouds by 25 atoms per
  cloud, with common seeded clouds across detuning and the multilevel Doppler
  overlay. The deterministic restoring/damping sweep uses 111 detunings from
  -0.5 through -6.0 by -0.05, with gravity excluded.
- The completed September 2026 campaign is a post-audited hybrid dataset. The
  detuning audit replaced 225 endpoint-timeout classifications across 11 scan
  points, and the raw-saturation audit replaced 155 across 4 scan points, using
  documented 200 ms coarse/fine trajectory checks. At `s0 = 0.25`, two
  near-edge rays have genuine nonmonotone low-speed capture islands: disc 6,
  point 22 captures 0.50--1.25 m/s, and disc 17, point 23 captures 0.75--1.00
  m/s on the audited 0.25 m/s grid. Spectra and loading integrals must use
  those immutable velocity-resolved masks; their zero-valued scalar rows are
  compatibility fallbacks only. Preserve dataset revisions
  `83717017fca6aaa5fcff6986cb09bcff2df782d3814c8451f1210f26e83fc7f7`
  (detuning) and
  `3dfa82f82e0b3edf1a6cceb804e56460c8e117e07db07bfc4714e9465143c4d5`
  (raw saturation) and the associated audit provenance.
- Every September 2026 temperature trajectory completed and remained inside
  the final 2 mm core, but all 23 aggregate detuning points fail the strict
  all-cloud stationarity gate (313 of 575 individual clouds pass). Report the
  plotted quantities as finite-25-ms final-window estimates, not equilibrium
  temperatures. A survivor fraction of one describes retention of preloaded
  clouds and is not an incident capture or loading fraction.
- The August 2026 sampling-disc-radius loading campaign uses 27 mW in each of
  the six cooling beams (the -15 MHz center-beam effective saturation is about
  one) and 0.1 mW in each repump beam. Its phase-one radii are 3, 5, 8, 12, 15,
  20, 25, and 30 mm; each radius uses 100 full-sphere incident-direction discs and
  100 independent uniform-area points per disc. Reuse one normalized seeded
  geometry across the eight radii so only the disc radius changes, then use an
  independent seed for the 100-by-100 confirmation run. Cross sections are
  direction-averaged projected areas with no 4-pi or octant multiplicity
  factor. Treat direction discs as the independent clusters for Student-t
  loading-rate and cross-section intervals.
- The multilevel temperature-sweep Doppler overlay is detuning dependent:
  `T_D = -hbar*Gamma^2/(8*k_B*Delta) *
  [1 + s_eff + (2*Delta/Gamma)^2]`, with angular-frequency `Delta < 0` and
  `Gamma`. Use the single-cooling-beam Gaussian-center convention
  `s_eff = s_0/[1 + (2*Delta/Gamma)^2]`, `s_0 = I_0/I_sat`, and recompute it
  at every detuning. The constant `hbar*Gamma/(2*k_B)` line belongs only to
  the simplified benchmark/special low-saturation point.
- The retained event-driven photon-jump engine is not the production engine;
  use it for short cross-checks of the rate approximation and internal-state
  dynamics.
- The replacement kernel passes its analytic two-level limit,
  transition-coefficient scaling, population conservation/positivity, and
  basic three-axis force-symmetry checks. Quantitative capture/loading and
  temperature claims remain prohibited until trajectory timestep/duration
  convergence and representative event-engine comparisons are documented.

## Authoritative pMOT geometry-stage assumptions

- The pMOT has no anti-Helmholtz coils and no applied external magnetic field.
  Its configuration must not contain a coil object or call the conventional
  quadrupole-field evaluator. Shared magnetic-field code remains available to
  `mot_simple`, the `mot_error` archive, and the rebuilt `mot_multilevel`
  model.
- Retain the six 780 nm cooling and six repump traveling components on the
  Cartesian x, y, and z paths. The first pMOT geometry configuration records
  the current comparison baseline of 27 mW per cooling component and 0.1 mW per
  repump component. The archived diagnostic currently constructs them through
  `mot_error` to preserve the exact 780.232684 nm repump wavelength. Production
  pMOT work must instead use the future validated `mot_multilevel` builder.
- Use one configurable trapping-laser frequency with default wavelength
  1529.268881 nm. "One trapping beam" means one frequency/configuration routed
  into three Cartesian round-trip paths, not one spatial ray: each path has an
  incident and retroreflected component, for six trapping components total.
- On each path the incident waist center is at -10 mm and the retro waist center
  is at +10 mm, so the focal positions are separated by 20 mm. Do not describe
  this as unequal lens focal lengths.
- Incident and retro helicities are independent configuration values using the
  propagation-frame `sigma+`, `sigma-`, or `pi` convention. The canonical
  geometry check uses `sigma+` for both traveling directions, which makes their
  lab-frame vector contributions oppose one another.
- Until measured waist parameters are supplied, use the symmetric ideal
  Gaussian in-trap approximation on x, y, and z: 35 mm input diameter and
  80.3 mm focal length, giving about 2.234 micrometers waist radius at the
  default wavelength. The external horizontal and vertical optical mechanics
  are not asserted to be identical.
- Trapping power and its three-path split are not yet specified. Geometry-stage
  intensity outputs must therefore be labeled per watt incident on one path;
  do not promote the superseded two-tone 0.5 W value into the current design.
- Geometry QA uses the standing-wave-averaged incoherent envelope. It must show
  both the even unsigned intensity sum and the odd helicity/direction-weighted
  optical-spin factor `sum(s_helicity * I * k_hat)`, where the propagation-frame
  convention gives `s_helicity = -1` for `sigma+`, `+1` for `sigma-`, and zero
  for `pi`. The factor is not itself an effective magnetic field and must not be
  used for quantitative force claims.
- No production pMOT trajectory may run until the trapping power/path split,
  coherent-interference treatment, window/mirror polarization transformations,
  state-resolved Stark Hamiltonian for all 24 states, conservative force,
  trap-light scattering/heating, and quantization-axis behavior at the
  fictitious-field zero have been specified and validated.

## Provisional pMOT AC-Stark diagnostic boundary

- `data/raw/pmot/Arora_CCSD_Differential_Polarizabilities.csv` contains one
  differential scalar/vector/tensor triplet per wavelength. It does not contain
  separate level-resolved 5S and 5P polarizabilities and therefore cannot
  uniquely determine all 24 hyperfine-Zeeman state shifts, a conservative
  Stark force, or 1529-nm scattering/heating.
- A provisional transition-level diagnostic is allowed only when its outputs
  are labeled as such. It applies the scalar differential shift directly,
  rescales the vector shift from the stretched cycling reference with the
  transition Zeeman coefficient, and uses an excited-state tensor angular
  proxy. These are ansatzes outside the stretched reference, not a recovered
  Hamiltonian.
- For every trapping component evaluate the nonrelativistic three-dimensional
  atom-frame wavelength separately:
  `lambda_seen = lambda_lab / [1 - dot(k_hat, v)/c]`. Preload and interpolate
  the narrow polarizability table; do not extrapolate, clip, silently switch to
  the coarse full-range table, reread the CSV at every step, or grow an
  unbounded cache over continuous trajectory wavelengths.
- The pMOT effective detuning in angular-frequency units is
  `Delta_eff = Delta_L - delta_HFS - k_cooling_or_repump.v - DeltaE_AC/hbar`.
  The external Zeeman term is exactly zero. The trapping-light Doppler shift
  changes the wavelength used for the polarizability lookup; it is not added
  directly as another 780-nm Doppler term.
- The archived diagnostic used 780-nm cooling/repump propagation-frame path helicities
  fixed at `(x, y, z) = (sigma+, sigma+, sigma-)` for both incident and retro
  components. Its negative velocity-force Jacobian came from the invalid
  archived rate kernel and must be re-established with the rebuilt solver.
- For the intended 1529.268881-nm vector-only design point, where the scalar
  and tensor differential shifts cancel, the unique centered matched-path
  helicity tuple that is position restoring on x, y, and z for the current
  waist order is also `(sigma+, sigma+, sigma-)` on both the incident and
  retro components. The globally reversed tuple `(sigma-, sigma-, sigma+)`
  is position anti-restoring. The other six centered matched tuples are
  saddles; unmatched incident/retro tuples bias the fictitious field at the
  origin. These labels are propagation-frame targets at the atoms, not direct
  laboratory waveplate settings. They are historical `mot_error` diagnostic
  classifications, not validated design conclusions; repeat the full
  three-dimensional Jacobian analysis with the rebuilt solver.
- Do not use the reversed `(sigma-, sigma-, sigma+)` result from the full
  provisional total-shift sweep as a design recommendation. In that diagnostic
  the helicity-independent -16.339691 MHz central shift changed the nominal
  -15 MHz stretched-reference cooling detuning to +1.339691 MHz (blue), so
  every centered configuration was anti-damping. Helicity cannot correct that
  common detuning error: either enforce the intended scalar/tensor cancellation
  or otherwise keep the relevant cooling transitions effectively red.
- Existing provisional pMOT diagnostics may import the `mot_error`
  explicit-local-environment entry point solely for reproducibility. New or
  production pMOT dynamics must not use it. The rebuilt `mot_multilevel` public
  entry point will be integrated only after its conventional-MOT validation.
- Physical pMOT trapping power remains unspecified. The first diagnostic's
  approximately 38.294 mW/path value is only the stretched-reference power
  scale corresponding to a nominal 20 G/cm vector-gradient proxy. It is not an
  apparatus default or power recommendation.
- Diagnostic trajectories include the unchanged 780-nm cooling/repump
  radiation pressure and optional recoil diffusion plus gravity. They exclude
  conservative Stark-gradient force, 1529-nm scattering/heating/loss,
  coherent standing-wave structure, measured polarization transformations,
  and nonadiabatic dynamics at the optical-spin zero. They must not be used for
  capture, loading, temperature, or quantitative trapping claims.
- `notebooks/pmot/trajectory_sampling.ipynb` is the current interactive
  vector-only diagnostic trajectory entry point. Its standard shot is the
  clean axial launch `r0=(15,0,0) mm`, `v0=(-17,0,0) m/s`, with gravity on,
  recoil diffusion off, 25 ms duration, and 5 microsecond step. It exposes all
  18 cooling, repump, and trapping traveling-component propagation-frame
  polarizations independently, initialized to the archived candidate `++-`
  tuple for incident and retro paths. That tuple must be revalidated. Its 3D
  view must draw the shared 12.7-mm
  cooling/repump volumes and the six 1529-nm Gaussian 1/e^2 envelopes at true
  in-trap scale; the 35-mm trapping diameter is a pre-lens input and must not
  be drawn through the trap.
- The archived inherited rate kernel combines saturated per-transition rates,
  explicit reverse stimulated-population links, and a force based on the
  ground-population-weighted available absorption rate. This closure is now
  known to be physically invalid for the multilevel MOT. Preserve and label it
  only for exact comparison with historical work; do not extend it or use it
  for quantitative multilevel or pMOT campaigns.
- One-dimensional force zeros do not establish stability. Classify candidate
  pMOT equilibria using the full three-dimensional force Jacobian and
  distinguish position-restoring static zeros from dynamically stable trapping.

## Python environment and commands

The project virtual environment is:
`/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC`

For every Python-related command, use this interpreter explicitly:
`/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python`

Run the suite with:
`/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -m pytest`

Use SI units internally. Label unit conversions explicitly in plots, tables,
saved metadata, and public APIs. Preserve user work in a dirty working tree and
do not rewrite notebooks or generated results unless the task requires it.
