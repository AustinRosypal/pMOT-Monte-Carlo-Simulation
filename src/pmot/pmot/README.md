# pMOT model

This package owns the pseudo-magneto-optical-trap implementation. It retains
six cooling and six repump components on Cartesian paths, contains no
anti-Helmholtz coils or external magnetic field, and routes one configurable
1529.268881-nm trapping-laser frequency through three Cartesian round trips.
The six focused traveling components have waist centers at -10 mm and +10 mm
on every axis.

The retained 780-nm components are constructed by the authoritative
multilevel builder, including its 780.232684-nm repump wavelength; the pMOT
does not maintain a second copy of those beam definitions.

An explicitly provisional differential-transition Stark layer is now present.
It evaluates the trapping wavelength seen by a moving atom separately for all
six components, interpolates the supplied scalar/vector/tensor differential
polarizabilities, converts the total transition energy shift to frequency, and
subtracts that resonance shift in the authoritative 24-state cooling/repump
rate equations. It never supplies an external magnetic field.

This is **not** a unique 24-state Stark Hamiltonian or a quantitative pMOT
prediction. The CSV has only one differential rank triplet per wavelength,
not separate ground/excited level polarizabilities. The vector and tensor
extensions outside the stretched cycling reference are therefore declared
approximations. Conservative Stark forces and 1529-nm scattering/heating are
also absent.

## Current contents

- `configuration.py` owns the no-coil pMOT apparatus, retained 780 nm light,
  paths, notebook order, and apparatus summaries.
- `trapping_beams.py` owns the single-frequency, configurable-helicity ideal
  Gaussian trapping-light envelope and scalar/vector-intensity geometry.
- `geometry_validation.py` generates the axial and planar intensity checks.
  Geometry outputs remain normalized per watt launched on one path.
- `polarizability.py` owns scalar lookups plus a preloaded, vectorized,
  range-checked narrow-table interpolator for trajectory Doppler wavelengths.
- `ac_stark.py` owns the provisional scalar/vector/tensor transition shifts,
  three-dimensional trapping-beam Doppler calculation, absolute path-power
  scaling, path helicities, and stretched-reference gradient calibration.
- `stark_trajectories.py` owns the no-coil wrapper and fixed-step trajectory
  integration around the unchanged 24-state cooling/repump kernel.
- `stark_trajectory_study.py` prints the effective-detuning equation and saves
  absorption-force, trajectory, kernel-absorption-rate, and Stark-decomposition
  diagnostics.
- `helicity_sweep.py` exhaustively audits all 64 independent sigma+/sigma-
  incident/retro path-helicity choices in the full provisional
  scalar/vector/tensor ansatz at a nominal -15 MHz carrier. Its common central
  shift makes the stretched reference blue detuned, so its apparent reversed
  restoring sign is explicitly retained only as a diagnostic artifact—not as
  the design-helicity result. It records origin bias and force, complete 3x3
  position and velocity Jacobians, and force/Stark/optical-spin comparisons.
- `vector_only_helicity_study.py` is a separate ideal-magic audit that leaves
  the authoritative 780-nm cooling/repump beam helicities fixed and imposes
  scalar-plus-tensor cancellation by applying only the vector transition
  shift. It enumerates all 64 trapping-helicity labels and classifies the eight
  zero-bias matched cases. Its position force is induced by the 1529-nm vector
  shift through modified 780-nm scattering; it includes no direct 1529-nm
  radiation-pressure or conservative force and does not change default physics.
  For the present waist order it finds matched `++-` incident and retro
  trapping tuples uniquely position restoring while the fixed 780-nm light
  remains damping; matched `--+` is position anti-restoring.
- `vector_only.py` is the plotting-backend-free core of that corrected
  ideal-magic observable. Interactive notebooks should import this module or
  `vector_only_trajectories.py`, not the batch helicity-study driver.
- `vector_only_trajectories.py` provides the configurable diagnostic
  trajectory runner. All cooling, repump, and trapping traveling-component
  propagation-frame polarizations are independent; the defaults are the
  known-working matched `++-` tuple. Its standard shot starts at
  `(15, 0, 0)` mm with velocity `(-17, 0, 0)` m/s. The default dynamics are
  deterministic, include gravity, and apply the two-entry-or-five-ms core
  flag, but that flag remains a provisional diagnostic rather than validated
  pMOT capture evidence.
- `trajectory_plotting.py` draws the shared 12.7-mm cooling/repump paths and
  all six focused 1529-nm Gaussian envelopes at physical scale together with
  the trajectory and time histories. The interactive entry point is
  `notebooks/pmot/trajectory_sampling.ipynb`; it exposes every traveling
  component's propagation-frame polarization and writes optional CSV, JSON,
  and PNG outputs below the pMOT output tree.
- `preliminary_scattering.py` is an old scalar two-level exploration retained
  only for notebook reproducibility. It is not an alternative production
  solver.

Reproduce the geometry outputs with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.geometry_validation
```

Run the provisional 20-G/cm-equivalent diagnostic with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.stark_trajectory_study
```

Run the full-provisional, blue-centered 64-case diagnostic with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.helicity_sweep
```

Its tables and manifest are saved under
`outputs/statistics/pmot/helicity_sweep_20Gpcm`; its PNG figures are saved
under `outputs/figures/pmot/helicity_sweep_20Gpcm`.

Run the authoritative design-helicity audit for the intended ideal-magic
vector-only condition with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.pmot.vector_only_helicity_study
```

This preserves the full-provisional outputs above and writes to the separate
`outputs/statistics/pmot/vector_only_helicity_20Gpcm` and
`outputs/figures/pmot/vector_only_helicity_20Gpcm` roots.

Use `--power-mw-per-path` to replace the provisional gradient-derived power.
The default diagnostic uses 27 mW per cooling beam, 0.1 mW per repump beam, a
standing-wave-averaged trapping envelope, and deterministic recoil-free
external motion so force-law behavior is reproducible.

## Physics-construction boundary

The pMOT calls the public explicit-local-environment entry point of the
validated `pmot.mot_multilevel` rate kernel. This prevents duplication of the
cooling/repump population solver while all pMOT-specific optical environment,
Stark approximation, trajectory code, and outputs remain in this package.

The provisional layer demonstrates the detuning plumbing, but production still
requires separate level-resolved polarizabilities, hyperfine recoupling and
local Hamiltonian diagonalization, conservative trapping-light forces,
scattering/heating terms, and well-defined dynamics at the fictitious-field
zero. Do not expand `preliminary_scattering.py` for that work.

The inherited rate kernel reports ground-population-weighted available
absorption and applies its momentum as the force while also retaining explicit
stimulated-emission population links. That saturated-rate closure still needs
a solver-wide two-level-limit and event-engine validation. The pMOT diagnostic
preserves it exactly as requested and labels its force as an absorption-force
proxy; it does not treat the resulting force sign or magnitude as validated
trapping performance.

No provisional trajectory output may be promoted to capture, temperature,
loading-rate, or trapping-performance evidence until trapping power/path split,
optical coherence, polarization transformations, the hyperfine-resolved
Hamiltonian, conservative force, and trap-light scattering have been settled
and validated. See
[`docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md`](../../../docs/pmot/GEOMETRY_AND_TRAPPING_BEAMS.md)
and
[`docs/pmot/PROVISIONAL_AC_STARK_MODEL.md`](../../../docs/pmot/PROVISIONAL_AC_STARK_MODEL.md).
