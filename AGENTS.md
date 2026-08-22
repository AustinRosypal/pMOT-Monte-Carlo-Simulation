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
is isolated in `src/pmot/mot_multilevel` and follows `MULTILEVEL_MOT.md` using a
23-state, event-driven, no-repumper model. `src/pmot/mot` now contains only the
reusable anti-Helmholtz field implementation and its plots. The future repumper
extension must include all relevant dipole-allowed transitions, including
F=1 -> F'=0.

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
operation required in the later multilevel model.

## Capture-velocity convention

- Sample incident directions uniformly in solid angle in one symmetry octant.
- A sampling disc is perpendicular to its incident direction.
- All launch velocities on a disc are parallel to the disc normal; offset
  points do not individually aim at the origin.
- Sample disc points uniformly in area.
- Current early-exit trapped criterion: the atom enters the central 2 mm-radius
  core twice, with an intervening exit. This is deliberately inexpensive and
  must be convergence-checked against longer bounded-trajectory diagnostics.
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
- `src/pmot/mot_multilevel`: isolated replacement multilevel atomic structure,
  light coupling, Gillespie event, recoil, and validation layers.
- `src/pmot/mot`: reusable anti-Helmholtz field implementation and field plots;
  it no longer contains a multilevel state/scattering engine.
- `src/pmot/configuration.py`, `beams.py`, `fields.py`: shared apparatus and beam
  geometry.
- `data/raw/pmot`: differential-polarizability datasets for the later pMOT phase.
- `notebooks/mot_simple`: current interactive validation and sampling notebooks.
- `tests`: automated physics and numerical checks.
- `docs/shared/BFIELD.md`, `docs/mot_multilevel/ZEEMAN.md`, and
  `docs/mot_simple/SAMPLINGALGORITHM.md`: historical derivations and
  requirements; this file and explicit user decisions take precedence if they
  conflict.

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
