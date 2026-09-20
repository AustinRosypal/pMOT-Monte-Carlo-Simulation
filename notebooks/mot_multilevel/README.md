# Multilevel MOT rebuild notebooks

These ipywidgets notebooks use only the replacement Section-12 population-rate
solver. They initialize cooling power to 27 mW per traveling beam:

- `trajectory_explorer.ipynb`: configure a single launch, cooling/repump powers,
  detuning, coil gradient, and integration settings; inspect beam geometry,
  motion, net beam rates, optical force, and manifold populations.
- `trajectory_animation.ipynb`: animate a configured trajectory and all 24
  time-varying state populations; optionally save both GIFs.
- `capture_loading_explorer.ipynb`: sample capture velocities, cross sections,
  and loading rates while controlling the coil gradient and launch geometry.
  Name each run to preserve outputs; a matching run can be resumed after
  interruption. Its small defaults are diagnostic, not converged results.
  Loading uses the project's fixed reference vapor distribution.

Historical notebooks are retained in `notebooks/mot_error` for provenance only.
New outputs are separated under `mot_multilevel_population_rate_v1`.
