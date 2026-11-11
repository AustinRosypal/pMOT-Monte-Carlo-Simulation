# Model comparison notebooks

This is the only directory for cross-model comparison notebooks.

- [force_and_gradient_multilevel_mot_strange_mot_2level_pmot_20261020.ipynb](force_and_gradient_multilevel_mot_strange_mot_2level_pmot_20261020.ipynb) — Multilevel MOT, strange MOT, and effective two-level pMOT force and spatial-gradient line cuts, including transverse offsets.
- [force_and_gradient_multilevel_mot_strange_mot_2level_pmot_svt_vs_vector.ipynb](force_and_gradient_multilevel_mot_strange_mot_2level_pmot_svt_vs_vector.ipynb) — Current force and spatial-gradient overlays; compares multilevel MOT, strange MOT, and scalar/vector/tensor versus vector-only effective two-level pMOT.
- [loading_2level_mot_vs_multilevel_mot_exact_d2_20261104.ipynb](loading_2level_mot_vs_multilevel_mot_exact_d2_20261104.ipynb) — Original exact-D2 two-level versus physical multilevel MOT loading scans: saturation, effective saturation, detuning, gradient, and diameter.
- [loading_2level_mot_vs_multilevel_mot_exact_d2_refined_20261104.ipynb](loading_2level_mot_vs_multilevel_mot_exact_d2_refined_20261104.ipynb) — The same two-model loading comparison using the independently planned, refined-sampling multilevel campaign.
- [loading_gradient_detuning_diameter_2level_mot_multilevel_mot_strange_mot_20261111.ipynb](loading_gradient_detuning_diameter_2level_mot_multilevel_mot_strange_mot_20261111.ipynb) — Two-level MOT, physical multilevel MOT, and strange MOT loading versus gradient (both geometries), detuning, and diameter.
- [loading_gradient_2level_mot_multilevel_mot_strange_mot_2level_pmot_20261111.ipynb](loading_gradient_2level_mot_multilevel_mot_strange_mot_2level_pmot_20261111.ipynb) — The preceding three MOTs plus effective two-level pMOT loading versus gradient; both geometries, both exact 1529-nm wavelengths, and svt/sv/v Stark components.

## Output and naming convention

Comparison figures and associated exports belong under
`outputs/figures/model_comparisons/<descriptive-study-name>`. Import canonical
paths and notebook names from `pmot.comparison_paths`. Notebook names identify
what is plotted, the compared models, and the scan or variant; include a campaign
date when applicable. Do not recreate singular comparison folders or a separate
MOT comparison directory.

The loading extension figures through 40 G/cm are under
`outputs/figures/model_comparisons/extended_loading_40gcm_20261111`.
The two exact-D2 benchmark figure sets retain their campaign-named subdirectories
under that same canonical figure root.

Simulation datasets and immutable audit records stay in their original campaign
roots. `relocation_20261111.json` maps historical presentation paths to current
paths and records file checksums. Older audit records can therefore retain their
original paths and checksums without being rewritten as if they were new audits.

## Force diagnostic controls

The force notebooks evaluate stationary atoms with gravity excluded. The pMOT
branches are provisional effective two-level diagnostics, not state-resolved
physical pMOT predictions. The current four-model notebook exposes the physical
powers, detuning, scan resolution, recomputation switch, and metadata.

For arbitrary scalar/vector/tensor selections, use the live browser comparison
under `visualizations/pmot_interactive` or:

```bash
uv run python scripts/plot_updated_force_gradient_overlays.py --pmot-components svt sv
```

The script labels component-aware curves `2Level_pMOT_<components>` and keeps
`pMOT_2Level` / `pmot_vec_2Level` as aliases for `2Level_pMOT_svt` /
`2Level_pMOT_v` for existing notebook callers.
