# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into three model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_multilevel`: Section-12 physical 24-state population-rate MOT kernel.
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
