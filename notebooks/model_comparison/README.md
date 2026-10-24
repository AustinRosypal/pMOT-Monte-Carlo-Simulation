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
