# Multilevel MOT rebuild notebooks

These ipywidgets notebooks use only the replacement Section-12 population-rate
solver. They initialize cooling power to 27 mW per traveling beam:

- `trajectory_explorer.ipynb`: configure a single launch, cooling/repump powers,
  detuning, coil gradient, and integration settings; inspect beam geometry,
  motion, net beam rates, optical force, and manifold populations.
- `trajectory_animation.ipynb`: animate a configured trajectory and all 24
  time-varying state populations; optionally save both GIFs.
- `configured_trajectory_lab.ipynb`: define the cooling, repump, beam-size,
  coil-gradient, numerical, and launch parameters in one reproducible place;
  preview true-scale beam envelopes and magnetic-field planes; run either one
  manual launch or a seeded full-sphere trajectory ensemble; inspect selected
  3D paths, time diagnostics, and an inline trajectory animation.
- `capture_loading_explorer.ipynb`: sample capture velocities, cross sections,
  and loading rates while controlling the coil gradient and launch geometry.
  Name each run to preserve outputs; a matching run can be resumed after
  interruption. Its small defaults are diagnostic, not converged results.
  Loading uses the project's fixed reference vapor distribution.

Historical invalid-solver notebooks were removed. New outputs are separated
under `mot_multilevel_population_rate_v1`.
