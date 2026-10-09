# Multilevel MOT rebuild notebooks

These notebooks use only the replacement Section-12 population-rate solver.
They initialize cooling power to 27 mW per traveling beam:

- `trajectory_explorer.ipynb`: configure a single launch, cooling/repump powers,
  detuning, coil gradient, and integration settings; inspect beam geometry,
  motion, net beam rates, optical force, and manifold populations.
- `trajectory_animation.ipynb`: animate a configured trajectory and all 24
  time-varying state populations; optionally save both GIFs.
- `configured_trajectory_lab.ipynb`: define the cooling, repump, beam-size,
  coil-gradient, numerical, and launch parameters in one reproducible place;
  preview true-scale beam envelopes and magnetic-field planes; run either one
  manual Cartesian launch, one direction-disc launch with specified polar and
  azimuthal angles and impact parameter, or a seeded full-sphere trajectory
  ensemble; inspect selected 3D paths, time diagnostics, and an inline
  trajectory animation.
- `mot_with_pmot_fields_trajectory_lab.ipynb`: use the same configurable
  trajectory workflow with the physical 24-state multilevel MOT, but replace
  the anti-Helmholtz coil with the intended pMOT trapping-intensity envelopes
  mapped to a centered surrogate magnetic field. Configure the MOT,
  surrogate-field geometry, launch, integration, saving, and coarse-grained
  full-trajectory animation through ordinary dictionaries. This is a MOT with
  pMOT-shaped fields, not a physical pMOT prediction.
- `capture_loading_explorer.ipynb`: sample capture velocities, cross sections,
  and loading rates from editable `PHYSICS` and `CAPTURE` dictionaries. Run it
  cell-by-cell, run all cells, or execute it headlessly with `jupyter
  nbconvert`. Name each run to preserve outputs; a matching run can be resumed
  after interruption. Its small defaults are diagnostic, not converged
  results. Loading uses the project's fixed reference vapor distribution.

Historical invalid-solver notebooks were removed. New outputs are separated
under `mot_multilevel_population_rate_v1`.
