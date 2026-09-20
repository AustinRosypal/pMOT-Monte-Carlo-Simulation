# Physical multilevel Rb-87 MOT

This package implements the Section 12 population-rate model specified in
`PopulationRateEq_Instructions.md` for the 24 hyperfine-Zeeman states of the
Rb-87 D2 line.

The former implementation is preserved in `src/pmot/mot_error`. It must not be
used for physical MOT or pMOT claims because it inserted a saturated two-level
scattering expression into a multilevel population-rate matrix.

## Solver flow

`atomic_structure.py` calls ARC once through a cached builder. It precomputes:

- the 8 ground and 16 excited states;
- all 54 allowed hyperfine-Zeeman electric-dipole matrix elements;
- each zero-field transition frequency;
- each pairwise spontaneous rate
  `Gamma[g,e] = omega**3 |d[g,e]|**2 / (3*pi*epsilon_0*hbar*c**3)`;
- each excited-state total `Gamma_e = sum_g Gamma[g,e]`;
- exact weak-field hyperfine Lande factors from ARC.

At each local phase-space point, `rate_equations.py` evaluates the
anti-Helmholtz field and quantization axis, projects every beam polarization,
and calculates

```text
Omega[b,e,g] = -E_b epsilon[b,q] d[e,g] / hbar
Delta[b,e,g] = omega_laser - omega[e,g] - k_b.v - Delta_Z[e,g]
W[b,e,g] = Gamma_e |Omega[b,e,g]|^2 /
             (Gamma_e^2 + 4 Delta[b,e,g]^2)
```

There is deliberately no `1+s` term in `W`. The summed `W[e,g]` supplies
equal stimulated absorption and emission links. ARC-derived spontaneous rates
are added only from excited to ground states, and the diagonal is chosen so
every column of the 24x24 matrix sums to zero. The normalized steady state is
then solved from `M p = 0`.

The beam force uses the beam-resolved net laser photon rate,

```text
F_b = hbar k_b sum_e,g W[b,e,g] (p_g - p_e).
```

The deterministic trajectory driver uses a 5 microsecond default step and
recomputes the entire local population/force problem at every RK4 stage.

## Simulation and analysis scaffolding

The current default is 27 mW in each of the six cooling traveling beams,
0.1 mW in each repump component, and -15 MHz cooling detuning. The completed
September 19 ten-atom probe remains pinned to its original 20 mW/beam and
must not be relabeled as a 27 mW result.

- `diagnostics.py` provides trajectory summaries, pandas tables, CSV/NPZ/JSON
  persistence, 3D beam/trajectory plots, time-series diagnostics, and GIF or
  in-notebook trajectory and 24-bin population-histogram animations.
- `capture.py` provides seeded full-sphere launch discs, uniform-area impact
  points, deterministic RK4 capture classification, fail-closed
  trapped/escaped bracketing, capture-velocity searches, capture cross
  sections, and loading-rate quadrature. Long runs checkpoint sample results
  and can resume only with an exactly matching saved configuration.
- `notebooks/mot_multilevel/trajectory_explorer.ipynb` is the main ipywidgets
  trajectory, beam, population, scattering, and force explorer.
- `notebooks/mot_multilevel/trajectory_animation.ipynb` creates interactive
  trajectory and 24-state population animations and optional GIF files.
- `notebooks/mot_multilevel/capture_loading_explorer.ipynb` controls small
  capture-velocity, cross-section, and loading-rate studies. Its small defaults
  are for inspection, not production statistics.
- `scripts/run_ten_atom_population_campaign.py` runs a seeded ten-launch
  demonstration and saves individual/combined trajectories, a 24-state
  population GIF, capture brackets, an illustrative cross section and loading
  quadrature, and a three-axis restoring-force plot. Its companion
  `scripts/audit_ten_atom_population_campaign.py` checks selected outcomes and
  bracket endpoints at half the timestep and a longer observation window.
  Products live under the `ten_atom_section12_probe_20260919` output child.
- `scripts/run_25x20_multilevel_capture.py` runs the checkpointed 27 mW/beam
  25-direction-disc by 20-point full-sphere campaign (seed 20260921, 15 mm
  disc radius) at the default 5 microsecond RK4 step. All velocities on a
  disc are parallel to its inward normal. It saves geometry, 500 per-ray
  capture brackets, a 0--60 m/s cross-section spectrum, disc-cluster
  intervals, and the loading-rate quadrature under
  `capture_loading_25x20_r15mm_27mW_dt5us_seed20260921` in the statistics
  and figures output roots. The previous 20 microsecond output remains under
  `capture_loading_25x20_r15mm_27mW_seed20260921` and must not be relabeled
  as a 5 microsecond result. `scripts/audit_25x20_multilevel_capture.py` checks
  one selected ray per disc at half the trajectory step and twice the minimum
  observation time. The scalar-boundary spectrum remains provisional pending
  a full velocity-mask and all-ray convergence audit.

All generated products use `outputs/{trajectories,figures,statistics}/
mot_multilevel_population_rate_v1`. No new scaffolding imports `mot_error`.

## Status boundary

The core equations have automated tests for ARC constants, matrix dimensions,
column conservation, population normalization and positivity, steady-state
residuals, the analytic two-level limit, photon-flow balance, origin symmetry,
and three-axis damping/restoring signs. Capture/loading scaffolding is now
implemented, but its binary boundary assumption and every reported regime
still require velocity-mask, timestep, duration, and sampling convergence
before quantitative claims. A recoil-diffusion temperature layer remains
unimplemented.

New results must use the `mot_multilevel_population_rate_v1` output namespace.
Historical `outputs/.../mot_multilevel` data came from `mot_error` and remain
invalid as physical multilevel-MOT predictions.
