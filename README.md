# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into three model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_multilevel`: Physical 24-state population-rate MOT kernel.
- `pmot`: **MOT with pMOT Fields** studies plus future pMOT-development code.
  The highlighted runnable study keeps the physical multilevel MOT dynamics
  but replaces the coil field with a surrogate magnetic field shaped by the
  intended pMOT trapping-beam intensities.

## Repository layout

| Area | Two-level MOT | Physical multilevel MOT | MOT with pMOT Fields |
|---|---|---|---|
| Source | `src/pmot/mot_simple` | `src/pmot/mot_multilevel` | `src/pmot/pmot/surrogate_effective_field*.py` |
| Primary entry points | `notebooks/mot_simple` and CLI | `notebooks/mot_multilevel` | `notebooks/mot_multilevel/mot_with_pmot_fields_trajectory_lab.ipynb` and CLI |
| Tests | `tests/mot_simple` | `tests/mot_multilevel` | surrogate-field tests under `tests/pmot` |
| Documentation | `docs/mot_simple` | `docs/mot_multilevel` | `PMOT_EFFECTIVE_FIELD_CODEX.md` and `docs/pmot` |

The other modules and notebooks under `src/pmot/pmot` and `notebooks/pmot`
are retained for future pMOT development and provenance. They are not promoted
here as validated pMOT simulation entry points.

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

The physical multilevel workflow is exposed through notebooks under
`notebooks/mot_multilevel`. The trajectory explorers use ipywidgets; the
configured trajectory and capture/loading notebooks use ordinary editable
parameter dictionaries and can run top-to-bottom without UI controls. Their
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
full-sphere launches, an exact direction-disc launch with user-selected polar
and azimuthal angles and impact parameter, trajectory count, and output saving. Powers are per
traveling beam; detunings are entered in MHz; beam sizes are Gaussian
1/e-squared intensity diameters. Saved runs are written below
`outputs/trajectories/mot_multilevel_population_rate_v1/`.

For a capture-cross-section and loading-rate study, launch the
[capture/loading notebook](notebooks/mot_multilevel/capture_loading_explorer.ipynb):

```bash
uv run jupyter lab notebooks/mot_multilevel/capture_loading_explorer.ipynb
```

Edit the `PHYSICS` and `CAPTURE` dictionaries, then select **Run > Run All
Cells**. They expose the direction-disc count, points per disc, launch
geometry, capture-speed bracket and bisection, duration, timestep, capture
criterion, cross-section velocity grid, cooling and repump powers and
detunings, magnetic gradient and coil geometry, seed, worker count, checkpoint
interval, and run name. The number of sampled launch rays is `direction discs
x points per disc`, and every ray may require several trajectories while its
capture boundary is searched. This interface currently uses the standard
12.7-mm beam geometry.

After editing those same dictionaries, execute the notebook without opening
JupyterLab:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir outputs/notebook_runs \
  --output capture_loading_executed.ipynb \
  notebooks/mot_multilevel/capture_loading_explorer.ipynb
```

Use a new `run_name` for a new study. To continue an interrupted study, retain
all parameters and set `resume_matching_run` to `True`. Multilevel
capture/loading outputs remain diagnostic until the repository's full
convergence policy has been satisfied.

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

### MOT with pMOT Fields

This is the current bridge between the validated multilevel MOT and a future
physical pMOT. It answers a deliberately limited question: **if the spatially
varying field intended from the pMOT trapping-beam intensities acted as an
effective magnetic field, would an otherwise conventional MOT exhibit local
restoring force, velocity damping, and trapped trajectories?**

The simulation uses the physical Section-12 24-state population-rate MOT,
including the six 780-nm cooling beams, six repump beams, deterministic
mean radiation pressure, and gravity. It does not use an anti-Helmholtz coil.
Instead, the six intended 1529-nm intensity/helicity envelopes are mapped to a
centered, stretched-transition-equivalent surrogate magnetic field. That field
is supplied to the MOT Zeeman calculation at every local force evaluation and
RK4 stage.

#### Interactive trajectory notebook

Open
[`notebooks/mot_multilevel/mot_with_pmot_fields_trajectory_lab.ipynb`](notebooks/mot_multilevel/mot_with_pmot_fields_trajectory_lab.ipynb)
in VS Code, select the repository's `.venv` Python kernel, and choose **Run
All**. Edit the `PHYSICS`, `SURROGATE_FIELD`, and `SIMULATION` dictionaries near
the top before running to set the laser parameters, effective-field geometry,
initial atom state, timestep, duration, plots, animation, and output options.
The launch can be a manual Cartesian state, a direction-disc state with chosen
polar angle, azimuthal angle, and impact parameter, or a seeded random
full-sphere ensemble.

Alternatively, launch the notebook from the repository root in JupyterLab:

```bash
uv run jupyter lab notebooks/mot_multilevel/mot_with_pmot_fields_trajectory_lab.ipynb
```

If Jupyter prints a `localhost` URL but cannot open a browser automatically,
leave that terminal running and paste the printed URL into a browser. This is
only the notebook interface; the Python kernel still runs locally in the
repository environment. Notebook results can optionally be saved under
`outputs/trajectories/mot_with_pmot_fields/` and
`outputs/figures/mot_with_pmot_fields/`.

#### Complete diagnostic study

Run the complete field, force, trajectory, and timestep-audit study from the
repository root:

```bash
uv run python scripts/run_pmot_surrogate_effective_field_test.py --stage all --output-directory "outputs/diagnostics/pmot/MOT with pMOT Fields/example"
```

The stages can also be run individually with `--stage field`, `--stage force`,
`--stage trajectories`, or `--stage audit`. After all required stages exist,
`--stage report` regenerates the summary report. Use `--help` to show the
available command-line arguments:

```bash
uv run python scripts/run_pmot_surrogate_effective_field_test.py --help
```

The selected output directory contains:

- `README.md` and `campaign_manifest.json`: verdict, evidence gates, and run
  provenance;
- `figures/effective_field`: field-axis cuts, planes, and three-dimensional
  vector plots;
- `figures/force`: restoring and damping comparisons against a conventional
  10 G/cm coil MOT;
- `figures/trajectories`: surrogate-field and coil-MOT trajectory comparisons;
- `data/`: the sampled fields, force diagnostics, trajectories, and timestep
  audit in machine-readable form.

The default surrogate-field definition is in
[`src/pmot/pmot/surrogate_effective_field.py`](src/pmot/pmot/surrogate_effective_field.py),
and the diagnostic trajectory cases and analysis are in
[`src/pmot/pmot/surrogate_effective_field_study.py`](src/pmot/pmot/surrogate_effective_field_study.py).

#### Longer capture and loading survey

The repository also contains the completed campaign definition for 25
full-sphere direction discs, 10 uniform-area points per disc, a 15-mm sampling
disc, and a direct 1--100 m/s velocity mask:

```bash
uv run python scripts/run_pmot_centered_surrogate_capture_loading.py
```

This launches 5,500 deterministic trajectories, uses eight worker processes,
and permits up to 12 hours of wall time. It checkpoints and resumes in:

```text
outputs/diagnostics/pmot/MOT testing with a strange defined magnetic field/centered field full-sphere capture loading 25x10 r15mm 20261002/
```

Its fixed sampling, velocity, timestep, duration, and worker parameters are
defined near the top of
[`scripts/run_pmot_centered_surrogate_capture_loading.py`](scripts/run_pmot_centered_surrogate_capture_loading.py).

#### Interpretation boundary

Despite the pMOT-shaped field, this is a **MOT with pMOT Fields**, not a
physical pMOT simulation. The trapping light is represented only through a
surrogate magnetic field. The calculation does not yet include the unique
state-resolved scalar/vector/tensor AC-Stark Hamiltonian, conservative
trapping-light force, trapping-light scattering or heating, coherent
interference, measured polarization transformations, or nonadiabatic dynamics
at the fictitious-field zero. Results test the intended field geometry inside
the conventional MOT dynamics; they must not be presented as quantitative
pMOT predictions.
