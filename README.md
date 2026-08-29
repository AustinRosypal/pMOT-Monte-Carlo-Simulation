# pMOT Monte Carlo

Monte Carlo and rate-equation simulations for Rb-87 laser cooling and trapping.
Development is separated into three model branches:

- `mot_simple`: validated deterministic effective two-level MOT.
- `mot_multilevel`: physical hyperfine/Zeeman MOT with the efficient population-rate engine.
- `pmot`: future pseudo-MOT trapping-light model; preliminary optical notebooks and data live here.

## Repository layout

| Area | Two-level MOT | Multilevel MOT | Future pMOT |
|---|---|---|---|
| Source | `src/pmot/mot_simple` | `src/pmot/mot_multilevel` | `src/pmot/pmot` |
| Notebooks | `notebooks/mot_simple` | `notebooks/mot_multilevel` | `notebooks/pmot` |
| Tests | `tests/mot_simple` | `tests/mot_multilevel` | `tests/pmot` |
| Processed data | `data/processed/mot_simple` | `data/processed/mot_multilevel` | `data/processed/pmot` |
| Outputs | `outputs/<category>/mot_simple` | `outputs/<category>/mot_multilevel` | `outputs/<category>/pmot` |
| Documentation | `docs/mot_simple` | `docs/mot_multilevel` | `docs/pmot` |

Reusable apparatus, beam, anti-Helmholtz-field, launch-disc, capture-analysis,
loading-rate, and plotting primitives remain directly under `src/pmot`;
model-specific algorithms must stay in their model branch. There is no generic
`mot` package: production implementations are explicitly `mot_simple`,
`mot_multilevel`, or the future `pmot` branch.
Historical preliminary results are retained under `pmot/legacy_preliminary`
directories rather than mixed with current MOT results.

Use the project environment for every Python command:

```bash
/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python -m pytest
```
