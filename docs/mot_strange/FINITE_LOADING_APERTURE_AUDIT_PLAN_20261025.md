# Remaining aperture and launch-placement checks

This is a proposed companion-audit design, not a convergence certificate or a
frozen trajectory allocation. Do not change the live production collector or
start a competing worker pool. Preserve its 15 mm aperture, 10 mm plane-center
distance, fixed 3,975-disc allocation and source snapshot.

The prospective geometry helpers are implemented separately in
`scripts/mot_strange_geometry_audit_sampling.py`. Four sampling tests pass in
`tests/mot_strange/test_geometry_audit_sampling.py`, covering exact position
reuse, uniform-area radial moments/CDFs with and without clipping, equal-radius
reuse, and rejection of old-plane points at a displaced plane. These helpers
run no trajectories and do not certify convergence or freeze an audit count.
They leave all frozen physical source and live collector files unchanged.

The full refined pilot's descriptive impact diagnostic is saved as
`validation/pilot_aperture_diagnostic.json` in the `_core_by2s_v2_dv0p125` root.
Central capture appears on three sampled rays between 12 and 15 mm impact,
with maximum captured impact 13.4967 mm. Their combined loading contribution
is 0.9400 atoms/s at the retained thermal inputs. This does not establish an
upper bound beyond 15 mm, or convergence with the launch plane. Never infer
zero missed loading from an empty sampled impact bin.

## Larger-radius comparison

A candidate next radius is 20 mm, with the plane center still 10 mm from the
origin. For each existing direction, calculate exact clipped areas A15 and
A20. The intersections are nested because the plane is unchanged.

An efficient paired design keeps exactly 20 accepted points on each larger
disc. Independently for each point, reuse its original 15 mm ray with
probability A15/A20; otherwise draw a uniform-area point from the clipped
15--20 mm annulus and evaluate a new trajectory. Annular proposals use uniform
angle and radius squared, reject outside-cell points, and retry. The mixture
is uniform on the larger disc/cell intersection. Weight the larger-disc mask
by A20/20, and the original by A15/20. Record all mixture selections and seeds.
Reuse a mask only for an exactly identical position, direction, speed, physical
configuration and observation definition. If A20=A15, the entire comparison
is identical without new rays.

Retain per-disc paired loading differences and spectra, not two independent
error bars. Report a paired interval and a separately explicit finite-sample
bound when the newly sampled annulus is all empty. Do not report an all-empty
annulus as an exact zero contribution.

## Launch-plane comparison

A candidate upstream distance is 12 mm, with radius still 15 mm. Generate
inside-cell uniform-area points and exact clipped areas at the new plane.
Common direction seeds and common underlying proposal streams can correlate
the comparison while preserving both marginal sampling distributions. Do not
translate old accepted points and merely reject invalid ones: the two clipped
intersections need not be nested in their transverse coordinates.

Every displaced-plane trajectory must be recomputed. Starting it at the old
plane with an unchanged velocity would omit optical force and gravity along
the added path. Keep velocities parallel to each disc normal; all starts must
remain inside the modeled glass cell.

## Allocation and acceptance

Use an independent paired pilot to determine fixed audit counts and freeze
the thresholds, seeds and counts before the comparison production. A candidate
numerical equivalence tolerance is 2% of reference loading, consistent with
the velocity-grid tolerance, with simultaneous coverage for the two geometry
checks. This proposal is not yet an executed or frozen audit. Preserve a
failed precision or equivalence gate; do not enlarge an ordinary sample until
its interval happens to pass.

`scripts/mot_strange_geometry_audit_statistics.py` provides prospective paired
statistics and fixed-count planning. Four tests pass for covariance-aware
differences, an effect outside tolerance, allocation limits and invalid/empty
reference samples. Relative equivalence uses the paired linear contrasts
`comparison - (1+tolerance)*reference` and
`comparison - (1-tolerance)*reference`, accounting for reference uncertainty.
Bonferroni across two geometry comparisons gives a proposed family
false-acceptance level <=5% under the Student-t assumptions. A passing paired
test alone does not certify numerical classifications, retention or the
separate omitted-phase-space bound. The helper explicitly leaves full
acceptance false and its proposed allocation unfrozen. No geometry comparison
trajectories or validation allocation have yet been generated.

Use the same 2 s horizon, unchanged core rule, explicit retained-at-horizon
outcome, dual-setting/fixed-edge checks, source provenance, thermal inputs and
per-disc spectra as the primary campaign. Include off-core retention separately.
Document the final radius/placement comparisons and their statistical limits,
and inspect final plots. The primary production calculation alone does not
prove a total-cell loading rate or discharge these remaining gates.
