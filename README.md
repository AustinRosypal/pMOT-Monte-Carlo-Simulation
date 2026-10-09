# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into three model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_multilevel`: Physical 24-state population-rate MOT kernel.
- `pmot`: provisional pseudo-MOT work using the rebuilt multilevel dissipative
  kernel; its Stark layer is not yet a physical pMOT prediction.

## Repository layout

| Area | Two-level MOT | Physical multilevel MOT | Provisional pMOT |
|---|---|---|---|
| Source | `src/pmot/mot_simple` | `src/pmot/mot_multilevel` | `src/pmot/pmot` |
| Notebooks | `notebooks/mot_simple` | `notebooks/mot_multilevel` | `notebooks/pmot` |
| Tests | `tests/mot_simple` | `tests/mot_multilevel` | `tests/pmot` |
| Documentation | `docs/mot_simple` | `docs/mot_multilevel` | `docs/pmot` |

Reusable apparatus, beam, anti-Helmholtz-field, launch-disc, capture-analysis,
loading-rate, and plotting primitives remain directly under `src/pmot`;
model-specific algorithms must stay in their model branch. There is no generic
`mot` package: implementations are explicitly `mot_simple`, `mot_multilevel`,
or `pmot`. Historical `outputs/<category>/mot_multilevel` and
`data/processed/mot_multilevel` names are retained as provenance for the
superseded invalid solver. New solver results are reserved under
`outputs/<category>/mot_multilevel_population_rate_v1`.
Historical preliminary results are retained under `pmot/legacy_preliminary`
directories rather than mixed with current MOT results.

The physical multilevel workflow is exposed through three ipywidgets
notebooks under `notebooks/mot_multilevel`: `trajectory_explorer.ipynb`,
`trajectory_animation.ipynb`, and `capture_loading_explorer.ipynb`. Their
trajectory, animation, capture-cross-section, and loading-rate support code is
in `src/pmot/mot_multilevel/diagnostics.py` and
`src/pmot/mot_multilevel/capture.py`. These tools write only to the new
`mot_multilevel_population_rate_v1` output namespace.

## Reproducible setup

Python 3.12 is the tested interpreter. Virtual environments are local machine
artifacts and must be recreated after cloning; do not copy a virtual
environment from another checkout.

The recommended setup uses [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync --all-extras
uv run python -m pytest
```

`uv` creates and manages the repository-local `.venv` automatically. The
checked-in `uv.lock` fixes the complete cross-platform dependency resolution.
Use `uv run python ...`, `uv run jupyter lab`, and `uv run python -m pytest`
without referring to an absolute interpreter path.

For a standard `venv`/`pip` installation on Linux, macOS, or WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

On native Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

The root `requirements.txt` is exported from `uv.lock` with all project extras
for users who do not use `uv`. Run commands from the repository root so
repository-relative input and output paths resolve consistently.

## Run simulations manually

Run every command below from the repository root. With `uv`, no environment
activation or machine-specific Python path is needed.

### Conventional multilevel MOT

For freely configurable 24-state MOT trajectories, launch the
[configured trajectory notebook](notebooks/mot_multilevel/configured_trajectory_lab.ipynb):

```bash
uv run jupyter lab notebooks/mot_multilevel/configured_trajectory_lab.ipynb
```

Edit the `PHYSICS` and `SIMULATION` dictionaries near the top of the notebook,
then select **Run > Run All Cells**. They expose cooling and repump powers and
detunings, cooling and repump beam diameters, the anti-Helmholtz axial gradient
and coil geometry, gravity, trajectory duration and timestep, manual or seeded
full-sphere launches, trajectory count, and output saving. Powers are per
traveling beam; detunings are entered in MHz; beam sizes are Gaussian
1/e-squared intensity diameters. Saved runs are written below
`outputs/trajectories/mot_multilevel_population_rate_v1/`.

For a capture-cross-section and loading-rate study, launch the
[capture/loading notebook](notebooks/mot_multilevel/capture_loading_explorer.ipynb):

```bash
uv run jupyter lab notebooks/mot_multilevel/capture_loading_explorer.ipynb
```

Set the direction-disc count, points per disc, launch geometry, velocity-search
controls, duration, timestep, cooling detuning, cooling and repump powers,
magnetic gradient, seed, worker count, and run name; then press **Run
capture/loading**. The number of sampled launch rays is `direction discs x
points per disc`, and every ray may require several trajectories while its
capture boundary is searched. This interface currently uses the standard
12.7-mm beam geometry. Multilevel capture/loading outputs remain diagnostic
until the repository's full convergence policy has been satisfied.

### Validated two-level MOT command line

The effective two-level capture sampler is directly scriptable. Show all
options with:

```bash
uv run python -m pmot.mot_simple.sampling --help
```

For example:

```bash
uv run python -m pmot.mot_simple.sampling --disc-count 25 --points-per-disc 25 --radial-distance-mm 15 --disc-radius-mm 15 --initial-velocity-guess 10 --velocity-tolerance 0.25 --max-simulation-time-ms 200 --time-step-us 5 --seed 20261009 --output-dir outputs/statistics/mot_simple/example --figures-dir outputs/figures/mot_simple/example
```

Calculate the loading rate from that run with:

```bash
uv run python -m pmot.mot_simple.loading --spectrum-csv outputs/statistics/mot_simple/example/capture_velocity_spectrum.csv --output-json outputs/statistics/mot_simple/example/loading_rate_result.json
```

The sampler's command-line options control sampling and numerics; its physical
defaults are the documented 12.7-mm, 20-mW-per-beam, -15-MHz, 10-G/cm
two-level MOT.

### Provisional pMOT trajectory diagnostic

Launch the interactive
[no-coil pMOT trajectory laboratory](notebooks/pmot/trajectory_sampling.ipynb)
with:

```bash
uv run jupyter lab notebooks/pmot/trajectory_sampling.ipynb
```

Run all notebook cells to create the control panel, choose a preset or edit the
launch, integration, cooling/repump, 1529-nm geometry/power, and component
polarization controls, then press **Run pMOT trajectory**. Merely opening the
notebook or running its cells does not start a trajectory. Enable **save CSV,
JSON, and PNG** to write results below
`outputs/trajectories/pmot/vector_only_local_axis_notebook/`.

This pMOT notebook is an explicitly provisional ideal-magic, vector-only
diagnostic. It omits the state-resolved scalar/vector/tensor Stark Hamiltonian,
conservative trapping-light force, trapping-light scattering/heating, coherent
interference, measured polarization transformations, and nonadiabatic dynamics
at the fictitious-field zero. Its trajectories must not be reported as
quantitative pMOT capture, loading, temperature, or trapping predictions.
