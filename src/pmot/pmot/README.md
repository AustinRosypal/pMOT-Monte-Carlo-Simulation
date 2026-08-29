# Future pMOT model

This package owns the future pseudo-magneto-optical-trap implementation. The
goal is to extend the validated 24-state, repumper-enabled population-rate MOT
by removing the anti-Helmholtz field and introducing spatially varying
scalar/vector/tensor AC Stark shifts from dedicated trapping light. The vector
shift is intended to provide the state-dependent, position-dependent role of a
magnetic-field gradient.

That physical model is **not implemented yet**. Nothing in this package should
currently be interpreted as a quantitative pMOT force, capture, temperature,
or loading-rate prediction.

## Current contents

- `configuration.py` owns pMOT paths, notebook order, and an apparatus summary.
- `polarizability.py` loads and interpolates the pMOT differential-
  polarizability tables and converts them to the repository's shift units.
- `preliminary_scattering.py` is an old scalar two-level exploration retained
  only for notebook reproducibility. It is not an alternative to
  `pmot.mot_simple` or `pmot.mot_multilevel`.
- `trajectories.py` and `plotting.py` provide preliminary notebook diagnostics
  and visualizations.

Reusable constants, Gaussian-beam primitives, the current 780 nm cooling and
repump geometry, atomic state containers, and magnetic-field utilities live at
the shared `pmot` package level. Production MOT physics remains isolated in
`pmot.mot_simple` and `pmot.mot_multilevel`.

## Construction boundary

The production pMOT must be built as an extension of `pmot.mot_multilevel`'s
24-state rate-equation architecture. Do not build it by expanding
`preliminary_scattering.py`. The new branch will need explicit trapping-beam
objects, a hyperfine-resolved AC Stark Hamiltonian, state-dependent transition
shifts, conservative trapping-light forces, scattering/heating terms, and
well-defined behavior at the fictitious-field zero.

The repository contains a historical two-tone 1529 nm concept, but it was
removed from executable configuration before multilevel development and is not
authoritative. Its exact recovered values, the inferred gradient mechanism,
the present 780 nm geometry, and the decisions still needed before construction
are documented in
[`docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md`](../../../docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md).

No production pMOT simulation should begin until those unresolved geometry,
power, polarization, atomic-Hamiltonian, and validation choices have been
settled and recorded.
