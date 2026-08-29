# Python package boundaries

The source tree has three explicit model packages:

- `mot_simple`: deterministic effective two-level MOT;
- `mot_multilevel`: authoritative 24-state, repumper-enabled rate-equation MOT;
- `pmot`: future pseudo-MOT branch and its currently preliminary optical tools.

These packages may use the model-neutral modules directly under this directory,
but `mot_simple` and `mot_multilevel` must not import from one another.

## Shared modules

- `atomic_data.py`: Rb-87 cooling and repump transition records used by more
  than one model.
- `beams.py`: Gaussian-beam mathematics and vector helpers.
- `configuration.py`: physical constants, cell/lens metadata, the common
  cooling/repump apparatus, and anti-Helmholtz coil configuration.
- `fields.py`: common MOT beam construction, filtering, local intensity, and
  model-neutral field-sampling diagnostics.
- `state.py`: external atom position/velocity record.
- `magnetic_fields.py` and `magnetic_field_plotting.py`: shared
  anti-Helmholtz calculation, sampling, and validation plots.
- `launch_geometry.py`: random solid-angle discs and uniform-area launch
  points, independent of any force model.
- `capture_statistics.py`: capture-threshold records, cross-section
  aggregation, persistence, and plots.
- `loading.py`: loading-rate integration from a capture cross section.
- `beam_plotting.py`: model-neutral beam-volume mesh construction.

## Model-owned code

Two-level force, RK4, validation, and command-line workflows stay in
`mot_simple`. Hyperfine structure, repumper coupling, population-rate forces,
Langevin temperature dynamics, and multilevel capture workflows stay in
`mot_multilevel`. Differential polarizability, future trapping-light geometry,
and pMOT-specific forces belong in `pmot`; its current
`preliminary_scattering.py` is retained only for exploratory notebook
reproducibility and is not a production solver.

There is intentionally no generic `mot` subpackage. Shared magnetic-field code
that formerly occupied that ambiguous name now lives in the shared modules
listed above.
