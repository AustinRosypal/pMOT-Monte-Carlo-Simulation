# Multilevel MOT package

The main full-MOT trajectory algorithm is the efficient adiabatic-elimination
population-rate model specified by `docs/mot_multilevel/EFFICIENT_MOT.md`. It retains the full
hyperfine/Zeeman structure, cooling light, and repumper without tracking
individual photon jumps. `pmot.mot_simple` remains the validated two-level
regression model.

The implemented state graph has 8 ground and 16 excited states. The extra
excited state relative to the 23-state cooling-only specification is `F'=0`,
which is retained because the authoritative repumper policy requires the
dipole-allowed `F=1 -> F'=0` channel.

Current layers:

- `configuration.py`: angular-frequency conventions, cooling, and repumper settings.
- `atomic_structure.py`: 24 indexed states and precomputed dipole/decay graphs.
- `polarization.py`: propagation-frame polarization and local spherical basis.
- `coupling.py`: Doppler, Zeeman, detuning, and laser-driven primitives.
- `rate_equations.py`: main steady-state population solve, mean force, recoil
  diffusion, and fixed-timestep Langevin trajectory.
- `rate_diagnostics.py`: CSV/NPZ persistence and trajectory/performance plots.
- `rate_run.py`: command-line runner for efficient full-MOT trajectories.
- `events.py`, `trajectory.py`, `simulation.py`: legacy Gillespie engine retained
  for short regression and visualization runs only.
- `jump_visualization.py`: isolated hyperfine transition-jump animation.

The rate-equation approximation includes populations only. It intentionally
does not model optical coherences, coherent dark states, or sub-Doppler
polarization-gradient cooling.

Run the focused efficient-model tests with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -m pytest -q \
  tests/test_multilevel_rate_equations.py
```

Run one 25 ms full-MOT atom at a fixed 5 microsecond timestep and save its raw
data, 3D trajectory, population, scattering, position, and velocity plots with:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.rate_run --duration-ms 25 --dt-us 5
```

Outputs are written under
`outputs/trajectories/mot_multilevel/rate_equation`.

Run the checkpointed 50-disc, 25-point capture-velocity and loading-rate study
with deterministic mean-force trajectories using:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.rate_capture --disc-count 50 --points-per-disc 25 --workers 8
```

The command reports every simulation number, checkpoints partial CSV data, and
resumes by default.  It saves sample-level capture brackets, the velocity
spectrum and capture cross section, per-disc plots, a capture heatmap, the
loading-rate integral, Monte Carlo uncertainty, and complete run metadata under
`outputs/statistics/mot_multilevel/loading_rate_50_discs_25_points`.

Run the 27 mW cooling/0.1 mW repump full-sphere sampling-disc-radius campaign
and its independently seeded confirmation with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python \
  -m pmot.mot_multilevel.sampling_disc_radius_study --workers 24
```

The phase-one radii are 5, 12, 15, 20, 25, and 30 mm. Each point uses 100
random full-sphere direction discs and 100 independent uniform-area impact
points per disc. The runner checkpoints every 100 completed capture-threshold
searches and resumes by default. It writes per-radius cross-section plots,
full-sphere geometry plots with the cooling beams, clustered loading-rate
statistics, the six-point loading-rate plot, and the convergence analysis below
`outputs/{statistics,figures}/mot_multilevel/` in the directory named
`loading_vs_sampling_disc_radius_full_sphere_100_discs_100_points_cooling27mW`.
The plotted 95% intervals treat each direction disc as one independent cluster;
the curve fit also uses the covariance created by reusing the same phase-one
directions across radii. If the saturating fit does not pass its goodness,
monotonicity, plateau, and convergence-radius uncertainty gates, the independent
confirmation uses the prescribed 12 mm fallback.

Run and visualize a configurable full-MOT launch directly in
`notebooks/mot_multilevel/full_mot_single_trajectory.ipynb`. It adapts the
`Single Trajectory` controls and 3D beam/path view from the simplified-MOT disc
geometry notebook to the efficient fixed-timestep population-rate engine. Each
run also displays Cartesian motion, per-beam scattering and force histories,
restoring/damping and effective one-dimensional potential curves, magnetic-field
component surfaces, and time-averaged hyperfine-manifold occupation percentages.

The expensive photon-jump animation remains available outside the main
algorithm for short pedagogical and regression runs:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python - <<'PY'
from pathlib import Path
from pmot.mot_multilevel.jump_visualization import create_hyperfine_jump_animation

path = Path("outputs/figures/mot_multilevel/hyperfine_jump.gif")
print(create_hyperfine_jump_animation(path))
PY
```

The legacy `diagnostics.py`, `screening.py`, and `performance.py` workflows
still exercise the event-driven model. They are not the production algorithm
for long trajectories or large ensembles.

Do not proceed to trusted capture/loading claims until the rate-equation model
has been checked for timestep convergence and benchmarked against short legacy
jump trajectories in representative regimes, particularly near `B=0`.
