# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into explicit model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_multilevel`: Physical 24-state population-rate MOT kernel.
- `mot_strange`: a conventional multilevel MOT whose coil field is replaced by
  a surrogate magnetic field shaped by the intended pMOT-beam intensities.
- `pMOT_2Level`: the effective two-level, no-coil pMOT model.
- `pmot_vec_2Level`: a controlled vector-only variant of `pMOT_2Level` that
  sets the differential scalar and tensor polarizabilities to zero.
- `pMOT_multilevel`: reserved placeholder for the future physical multilevel
  pMOT; no simulation is implemented there yet.

## Repository layout

| Area | Two-level MOT | Physical multilevel MOT | MOT with pMOT Fields | 2-Level pMOT | Vector-only 2-Level pMOT |
|---|---|---|---|---|---|
| Source | `src/pmot/mot_simple` | `src/pmot/mot_multilevel` | `src/pmot/mot_strange/surrogate_effective_field*.py` | `src/pmot/pMOT_2Level` | `src/pmot/pmot_vec_2Level` |
| Primary entry points | `notebooks/mot_simple` and CLI | `notebooks/mot_multilevel` | `notebooks/mot_strange/mot_strange_trajectory_lab.ipynb` and CLI | `notebooks/pMOT_2Level/trajectory_simulation.ipynb` and Python API | `notebooks/model_comparison/updated_force_gradient_overlays.ipynb`, Python API, and plotting script |
| Tests | `tests/mot_simple` | `tests/mot_multilevel` | surrogate-field tests under `tests/mot_strange` | `tests/pMOT_2Level` | `tests/pmot_vec_2Level` |
| Documentation | `docs/mot_simple` | `docs/mot_multilevel` | `MOT_STRANGE_EFFECTIVE_FIELD_CODEX.md` and `docs/pmot` | package docstrings and `docs/pmot` | `src/pmot/pmot_vec_2Level/README.md` |

The non-surrogate modules under `src/pmot/mot_strange` and the corresponding
notebooks are retained as provisional historical scaffolding and provenance.
Future physical multilevel-pMOT work belongs under `pMOT_multilevel`, not
`mot_strange`.

Reusable apparatus, beam, anti-Helmholtz-field, launch-disc, capture-analysis,
loading-rate, and plotting primitives remain directly under `src/pmot`;
model-specific algorithms must stay in their model branch. There is no generic
`mot` package: implementations are explicitly `mot_simple`, `mot_multilevel`,
`mot_strange`, `pMOT_2Level`, `pmot_vec_2Level`, or the future
`pMOT_multilevel`. Historical
`outputs/<category>/mot_multilevel` and
`data/processed/mot_multilevel` names are retained as provenance for the
superseded invalid solver. New solver results are reserved under
`outputs/<category>/mot_multilevel_population_rate_v1`.
Historical preliminary results are retained under `mot_strange/legacy_preliminary`
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

### Force and force-gradient model overlays

The comparison script recomputes the stationary-atom radiation-pressure force
and its spatial derivative from the live implementations of the conventional
multilevel MOT, `mot_strange`, the effective `pMOT_2Level`, and the controlled
vector-only `pmot_vec_2Level` variant. Gravity is excluded from these force-law
comparisons.

#### Interactive comparison notebook

The primary interactive plotting entry point is
[`notebooks/model_comparison/updated_force_gradient_overlays.ipynb`](notebooks/model_comparison/updated_force_gradient_overlays.ipynb).
Open that exact file from the repository explorer in VS Code or Codex, select
the repository's `.venv` Python kernel, and choose **Run All**. It can also be
opened from the repository root with:

```bash
uv run jupyter lab notebooks/model_comparison/updated_force_gradient_overlays.ipynb
```

The notebook is organized as a numbered workflow:

1. Section 1 locates the checkout and imports the live comparison code.
2. Section 2 contains all user controls. Edit the cooling detuning, multilevel
   cooling and repump powers, pMOT cooling power, three x/y/z trapping-path
   powers, scan range, sampling resolution, figure DPI, and output directory
   here before running.
3. Sections 3 and 4 construct the position grid and evaluate all four force
   models. This is the slow part of the notebook.
4. Section 5 checks the force and same-axis force gradient at the origin.
5. Sections 6--8 create the three-model overlay, complete four-model overlay,
   and full-versus-vector-only pMOT overlay, respectively.
6. Section 9 saves the selected settings and lists every generated artifact.

The first complete run should use `RECOMPUTE_PHYSICS = True`. After the
linecut CSV has been generated, set it to `False` to reuse those force samples
when redisplaying the plots. Set it back to `True` whenever a physical power,
detuning, scan range, or sampling setting is changed.

Notebook outputs are saved by default under:

```text
outputs/diagnostics/updated_force_gradient_overlays_notebook/
```

That directory receives the sampled force/gradient CSV, a JSON record of the
notebook controls, and these figures:

- `mot_strange_pmot2level_overlay.png`: MOT, strange MOT, and full two-level
  pMOT;
- `mot_strange_pmot2level_vector_overlay.png`: all four models, including the
  vector-only two-level pMOT;
- `pmot2level_full_vs_vector_only_overlay.png`: direct comparison of the full
  and vector-only two-level pMOT branches.

Every figure places force in the upper row and the numerical spatial derivative
of that same force curve in the lower row. These are stationary-atom,
gravity-free radiation-pressure diagnostics; they are not trajectory, capture,
loading, or temperature predictions.

#### Command-line comparison script

From the repository root, regenerate the three standard overlay figures with:

```bash
uv run python scripts/plot_updated_force_gradient_overlays.py
```

The default output directory is
`outputs/diagnostics/updated_force_gradient_overlays_20261023/`. It receives
the three PNG figures, the sampled force/gradient CSV, and a JSON file recording
all settings. The default figures compare (1) MOT, strange MOT, and full
two-level pMOT; (2) those three plus the vector-only two-level pMOT; and (3)
the full and vector-only two-level pMOT branches directly.

Show every control and its default with:

```bash
uv run python scripts/plot_updated_force_gradient_overlays.py --help
```

For example, this command chooses a new output directory, a +/-12-mm scan,
-20-MHz cooling detuning, and explicit pMOT cooling and trapping powers:

```bash
uv run python scripts/plot_updated_force_gradient_overlays.py --output-directory outputs/diagnostics/my_overlay --scan-limit-mm 12 --cooling-detuning-mhz -20 --pmot-cooling-power-mw 1.0571 --pmot-trapping-powers-mw 31.524 31.524 31.524
```

The multilevel MOT and strange MOT share the values selected by
`--multilevel-cooling-power-mw` and `--multilevel-repump-power-mw`. The full
and vector-only two-level pMOT branches share `--pmot-cooling-power-mw` and the
three x/y/z path powers supplied to `--pmot-trapping-powers-mw`. Use
`--scan-limit-mm` to change the symmetric spatial range. The
`--broad-samples`, `--center-samples`, and `--focus-samples` options control
the numerical resolution used both for the force curves and their directly
computed spatial derivatives. The default +/-12.5-mm range stays inside the
hard transverse-support boundary of the current strange-MOT surrogate field.

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
[`notebooks/mot_strange/mot_strange_trajectory_lab.ipynb`](notebooks/mot_strange/mot_strange_trajectory_lab.ipynb)
in VS Code, select the repository's `.venv` Python kernel, and choose **Run
All**. Edit the `PHYSICS`, `SURROGATE_FIELD`, and `SIMULATION` dictionaries near
the top before running to set the laser parameters, effective-field geometry,
initial atom state, timestep, duration, plots, animation, and output options.
The launch can be a manual Cartesian state, a direction-disc state with chosen
polar angle, azimuthal angle, and impact parameter, or a seeded random
full-sphere ensemble.

Alternatively, launch the notebook from the repository root in JupyterLab:

```bash
uv run jupyter lab notebooks/mot_strange/mot_strange_trajectory_lab.ipynb
```

If Jupyter prints a `localhost` URL but cannot open a browser automatically,
leave that terminal running and paste the printed URL into a browser. This is
only the notebook interface; the Python kernel still runs locally in the
repository environment. Notebook results can optionally be saved under
`outputs/trajectories/mot_strange/` and
`outputs/figures/mot_strange/`.

#### Complete diagnostic study

Run the complete field, force, trajectory, and timestep-audit study from the
repository root:

```bash
uv run python scripts/run_mot_strange_surrogate_effective_field_test.py --stage all --output-directory "outputs/diagnostics/mot_strange/MOT with pMOT Fields/example"
```

The stages can also be run individually with `--stage field`, `--stage force`,
`--stage trajectories`, or `--stage audit`. After all required stages exist,
`--stage report` regenerates the summary report. Use `--help` to show the
available command-line arguments:

```bash
uv run python scripts/run_mot_strange_surrogate_effective_field_test.py --help
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
[`src/pmot/mot_strange/surrogate_effective_field.py`](src/pmot/mot_strange/surrogate_effective_field.py),
and the diagnostic trajectory cases and analysis are in
[`src/pmot/mot_strange/surrogate_effective_field_study.py`](src/pmot/mot_strange/surrogate_effective_field_study.py).

#### Historical capture and loading survey — invalid after field-conversion correction

The saved 2026-10-02 `mot_strange` loading campaigns used a surrogate-field
conversion coefficient that was too large by exactly a factor of two. Their
trajectory classifications, cross sections, loading rates, and uncertainty
plots are retained for provenance only and must not be cited or regenerated as
current physical results. Because capture is a nonlinear binary outcome, the
old loading rates cannot be repaired by rescaling them.

The following command is therefore a historical runner, not an endorsed
production campaign:

```bash
uv run python scripts/run_mot_strange_centered_surrogate_capture_loading.py
```

It launched 5,500 deterministic trajectories, used eight worker processes,
and permitted up to 12 hours of wall time. Its invalid output is retained in:

```text
outputs/diagnostics/mot_strange/MOT testing with a strange defined magnetic field/centered field full-sphere capture loading 25x10 r15mm 20261002/
```

Do not resume that directory: its manifest intentionally prevents corrected
field parameters from being mixed with the old trajectories. A corrected
campaign must use a new run name, resolve every direct-mask classification with
dual-timestep and duration checks, and compute uncertainty across independent
direction discs separately at every parameter point. The historical fixed
sampling, velocity, timestep, duration, and worker parameters are defined near
the top of
[`scripts/run_mot_strange_centered_surrogate_capture_loading.py`](scripts/run_mot_strange_centered_surrogate_capture_loading.py).
The rerun acceptance gates and uncertainty equations are recorded in
[`docs/mot_strange/LOADING_RERUN_AND_UNCERTAINTY_AUDIT.md`](docs/mot_strange/LOADING_RERUN_AND_UNCERTAINTY_AUDIT.md).

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
