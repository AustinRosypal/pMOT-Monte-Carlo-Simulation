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
and visualization comparisons. `src/pmot/mot` contains only the reusable
anti-Helmholtz field implementation and its plots.

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

- Sample incident directions uniformly in solid angle in one symmetry octant.
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
- `src/pmot/mot`: reusable anti-Helmholtz field implementation and field plots;
  it no longer contains a multilevel state/scattering engine.
- `src/pmot/configuration.py`, `beams.py`, `fields.py`: shared apparatus and beam
  geometry.
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
