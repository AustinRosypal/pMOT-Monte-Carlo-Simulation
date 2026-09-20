# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into three model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_error`: archived multilevel work with an invalid two-level scattering-rate closure.
- `mot_multilevel`: Section-12 physical 24-state population-rate MOT kernel.
- `pmot`: provisional pseudo-MOT work; its inherited dissipative kernel currently comes from
  `mot_error` and is therefore not a physical pMOT prediction.

## Repository layout

| Area | Two-level MOT | Archived erroneous model | Physical multilevel MOT | Provisional pMOT |
|---|---|---|---|---|
| Source | `src/pmot/mot_simple` | `src/pmot/mot_error` | `src/pmot/mot_multilevel` | `src/pmot/pmot` |
| Notebooks | `notebooks/mot_simple` | `notebooks/mot_error` | `notebooks/mot_multilevel` | `notebooks/pmot` |
| Tests | `tests/mot_simple` | `tests/mot_error` | `tests/mot_multilevel` | `tests/pmot` |
| Documentation | `docs/mot_simple` | `docs/mot_error` | `docs/mot_multilevel` | `docs/pmot` |

Reusable apparatus, beam, anti-Helmholtz-field, launch-disc, capture-analysis,
loading-rate, and plotting primitives remain directly under `src/pmot`;
model-specific algorithms must stay in their model branch. There is no generic
`mot` package: implementations are explicitly `mot_simple`, `mot_error`,
`mot_multilevel`, or `pmot`. Historical `outputs/<category>/mot_multilevel`
and `data/processed/mot_multilevel` names are retained as provenance; they
refer to the archived erroneous model. New solver results are reserved under
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

Use the project environment for every Python command:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -m pytest
```
