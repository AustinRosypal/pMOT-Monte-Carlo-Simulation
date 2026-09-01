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
4. Remove the magnetic field, introduce the scalar/vector AC Stark shifts, and
   optimize trapping wavelengths, powers, and intensity gradients.

The obsolete preliminary multilevel attempt has been removed. The replacement
is isolated in `src/pmot/mot_multilevel`. Its authoritative production model is
the efficient 24-state, repumper-enabled, adiabatic population-rate-equation
MOT. `docs/mot_multilevel/EFFICIENT_MOT.md` defines the solver architecture;
`docs/mot_multilevel/REPUMPER.md`, this file, and the package README define the
24-state repumper transition graph. Any 23-state basis wording retained in
`EFFICIENT_MOT.md` is superseded by those sources.
It contains 8 ground states and 16 excited states; the extra excited state
relative to the original 23-state cooling-only specification is F'=0, retained
because the repumper includes every relevant dipole-allowed transition from
F=1. Long trajectories, capture/loading calculations, force sweeps, and
temperature calculations must use this rate-equation model. The event-driven
Gillespie implementation is retained only for short regression, diagnostic,
and visualization comparisons. Shared anti-Helmholtz calculations and plots
live in `src/pmot/magnetic_fields.py` and
`src/pmot/magnetic_field_plotting.py`; there is no generic `mot` package.

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
- `src/pmot/mot_multilevel`: authoritative 24-state, repumper-enabled
  population-rate MOT; it also retains isolated Gillespie event/recoil layers
  for short regression and diagnostic comparisons.
- `src/pmot/pmot`: future pMOT branch. Its current polarizability,
  visualization, and preliminary scattering utilities are not a production
  pMOT force model; see `docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md`.
- `src/pmot/configuration.py`, `beams.py`, `fields.py`: shared apparatus and beam
  geometry.
- `src/pmot/magnetic_fields.py`, `magnetic_field_plotting.py`,
  `launch_geometry.py`, `capture_statistics.py`, `loading.py`, `state.py`, and
  `beam_plotting.py`: model-neutral field, launch, capture-analysis, loading,
  state, and visualization primitives. Model packages may depend on these
  shared modules but must not depend on one another.
- `data/raw/pmot`: differential-polarizability datasets for the later pMOT phase.
- `notebooks/mot_simple`: current interactive validation and sampling notebooks.
- `tests`: automated physics and numerical checks.
- `docs/shared/BFIELD.md`, `docs/mot_multilevel/ZEEMAN.md`, and
  `docs/mot_simple/SAMPLINGALGORITHM.md`: historical derivations and
  requirements. `docs/mot_multilevel/MULTILEVEL_MOT.md` is the historical
  23-state, no-repumper specification. `docs/mot_multilevel/EFFICIENT_MOT.md`
  defines the production solver architecture, while
  `docs/mot_multilevel/REPUMPER.md` and
  `src/pmot/mot_multilevel/README.md` define the production repumper extension;
  together they supersede the historical specification as described above.
  This file and explicit user decisions take precedence if documents conflict.

## Authoritative multilevel MOT assumptions

- Atom: state-resolved Rb-87 D2 system with 8 ground and 16 excited states.
- Include cooling light and the repumper; the repumper transition graph must
  retain all relevant F=1 channels, including F=1 -> F'=0.
- Use the adiabatically eliminated population-rate equations for production
  mean force, capture/loading, and Langevin temperature trajectories.
- Use angular-frequency units consistently inside `mot_multilevel`.
- Recompute local intensities, Doppler and Zeeman shifts, polarization
  decomposition, state populations, scattering, and force at every required
  trajectory evaluation.
- Production capture/loading trajectories use deterministic multilevel mean
  force with recoil diffusion disabled so classifications are reproducible and
  can support the local-monotonicity assumption. Bracket and scan checks must
  still verify that assumption in representative regimes.
- Production temperature trajectories use the same multilevel rate-equation
  force with recoil diffusion enabled through the Langevin model.
- For the August 2026 relationship campaign, the completed raw-saturation and
  effective-saturation loading sweeps retain their 30 full-sphere direction
  discs by 30 uniform-area launch points per disc. The restarted detuning
  loading sweep uses 15 full-sphere direction discs by 15 launch points per
  disc, and the corresponding temperature sweep uses 15 independent preloaded
  clouds by 15 atoms per cloud. Treat the 15 direction discs or clouds as the
  independent clusters for Student-t intervals (14 degrees of freedom). Keep
  the 30x30 and 15x15 products in separately named output roots so their sample
  sizes and provenance cannot be confused.
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
- Quantitative multilevel force, capture/loading, and temperature claims remain
  provisional until the applicable force-grid or trajectory timestep/duration
  convergence checks and representative comparisons against the event-driven
  engine have been documented.

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
