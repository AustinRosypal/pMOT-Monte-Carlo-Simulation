# Model-comparison notebooks

`three_model_force_gradient_comparison.ipynb` reproduces the matched 2-by-3
force and same-axis force-gradient figures for the regular multilevel MOT,
`mot_strange`, and the provisional effective two-level pMOT. The notebook uses
stationary atoms and excludes gravity so the radiation-pressure curves can be
compared directly.

`updated_force_gradient_overlays.ipynb` is the current four-model comparison.
It uses the corrected `mot_strange` vector-to-effective-field conversion and
compares the conventional multilevel MOT, `mot_strange`, `pMOT_2Level`, and
`pmot_vec_2Level`. Its clearly separated sections expose the physical powers,
detuning, scan range, numerical resolution, recomputation switch, individual
overlay figures, origin checks, and output metadata.

For arbitrary scalar/vector/tensor pMOT comparisons, use the live browser
comparison under `visualizations/pmot_interactive` or the command-line script:

```bash
uv run python scripts/plot_updated_force_gradient_overlays.py --pmot-components svt sv
```

The component-aware script labels its curves `2Level_pMOT_<components>` and
retains `pMOT_2Level`/`pmot_vec_2Level` as backward-compatible aliases for
`2Level_pMOT_svt`/`2Level_pMOT_v` when the older notebook calls it.
