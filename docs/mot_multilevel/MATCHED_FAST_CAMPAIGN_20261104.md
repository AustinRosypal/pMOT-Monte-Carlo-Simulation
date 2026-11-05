# Accelerated physical multilevel MOT comparison

The user approved the combined adaptive-integration and independent-pilot
sampling speedup on November 4, 2026, targeting 12–36 hours with progress and
updated completion ETAs. All 124 physical settings and exact cooling wavelength
780.241209686 nm remain unchanged. The original fixed-step campaign, its 24,286
checkpoint files and its frozen source snapshot remain under their original
roots; no original rays enter the revised production confidence intervals.

The new campaign is `matched_two_level_exact_D2_fast_20261104`, under the same
`outputs/statistics/mot_multilevel_population_rate_v1`, physical-figure,
comparison-figure and notebook parent directories as its predecessor. Run from
the repository with `uv run python -u -m
pmot.mot_multilevel.matched_fast.campaign all --workers 20`. All Python commands
use `uv run python`.

The completed standalone HTML exports are in the
[MOT loading report catalog](../../visualizations/mot_loading/README.md).

The existing public physical Section-12 population-rate kernel supplies every
adaptive or fixed force stage. No state truncation, interpolation, force-law
change or historical invalid multilevel result is used. Adaptive DP5(4)
evaluations and fixed RK4 evaluations have separate, explicitly labelled audit
arrays. Every direct-mask capture edge and its neighbours, adaptive disagreement,
literal zero-speed node, positive scalar endpoints and longer retention check
receive the original fixed-step audit ladder. Positive-speed unresolved nodes
fail closed. A full held-out 321-node fixed velocity mask validates the hybrid
mask at every physical setting before production is authorized.

The independent pilot uses master direction discs 837–868, each with 16 original
area-uniform points. Held-out numerical validation uses discs 869–900. Both
sets are permanently excluded from revised production intervals. Baseline
production additionally excludes every direction containing an original
fixed-campaign checkpoint. Every production coordinate is an original master
coordinate. The two-level scalar samples and authoritative exceptional masks
are reanalyzed on exactly the same selected discs and points for comparison.
The full original two-level benchmark remains unchanged.

All per-setting production counts are fixed together after every pilot has
finished and before any revised production starts. Planning targets an 8%
relative 95% Student-t half-width with a 1.25 variance inflation factor, at
least 128 direction discs and at most 12,800 rays per setting. Choices use
16, 25, 32 or 50 points per disc and only available matched directions. If the
pilot cannot support the preferred precision within that predeclared budget,
the plan records the predicted limitation and the final report shows achieved
precision honestly. Zero observed pilot capture does not establish zero true
loading. No production result causes a CI-triggered extension or early stop.

Every ray is checkpointed and retained, with per-disc capture spectra and
fixed-pressure temperature reweighting. A fresh point-bound worker pool uses
parent-prepared deep-copied immutable controls, verifies the complete actual
array fingerprint and compares actual-worker forces/populations with the full
24-state reference. Every ray saves that fingerprint; complete runtime arrays
are saved per setting. Independent saved-node replays must pass before pilot
or production publication. `eta.json` updates from per-setting
pilot costs and actual completed rays; its completion range is a forecast,
not a reason to bypass numerical gates or stop unfinished work.

The first persistent-worker pilot run was quarantined after a saved low-gradient
trajectory failed to reproduce in independent fresh-process replays. No main
production existed, and none of those pilot results selected the revised sample
counts. All earlier pilot and validation evidence remains under the
`revisions/worker_state_reproducibility` roots. The underlying runtime cause was
not conclusively localized; fresh point-bound state and actual-input/reference
and replay gates provide containment and verification rather than a claim that
the original cause is proven. The force, integrator equations, capture rule and
strict timeout ladder are unchanged. All pilot and held-out validation cases
are repeated. See `runtime_input_isolation_revision.json` for signatures and
evidence.

## Future statistical extensions

New independent direction-disc batches can be appended without rerunning old
trajectories. Each batch has a unique identity, fixed counts declared before
its outcomes, matching physical controls, and retained per-disc masks. Pilot
and validation observations never enter production intervals.

If the complete combined sample size is fixed before production outcomes are
examined, ordinary combined direction-cluster analysis is appropriate. For
later extensions chosen after looking at earlier results,
`matched_fast.extensions.combine_independent_batches` provides a conservative
alternative: batch k spends alpha = 0.05/[k(k+1)], constructs its corresponding
Student-t interval, and combines the simultaneous batch bounds with direction
count weights. Total alpha is at most 0.05 over arbitrarily many extensions.
This retains the usual cluster Student-t assumptions and avoids claiming
unchanged fixed-sample 95% coverage after optional precision-based pooling.
The function rejects reused directions, pilots, changed controls and batches
whose counts were not fixed in advance. Additional points on already-used
discs require a separately specified cluster analysis and are not silently
treated as new independent discs.

The original local steady-state/incoherent-light and finite 15 mm aperture
limitations remain. These mean-force simulations predict no atomic temperature.
