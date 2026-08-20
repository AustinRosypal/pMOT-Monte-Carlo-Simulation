# Multilevel MOT package

The main full-MOT trajectory algorithm is the efficient adiabatic-elimination
population-rate model specified by `EFFICIENT_MOT.md`. It retains the full
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

Run and visualize a configurable full-MOT launch directly in
`notebooks/mot_multilevel/full_mot_single_trajectory.ipynb`. It adapts the
`Single Trajectory` controls and 3D beam/path view from the simplified-MOT disc
geometry notebook to the efficient fixed-timestep population-rate engine.

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
