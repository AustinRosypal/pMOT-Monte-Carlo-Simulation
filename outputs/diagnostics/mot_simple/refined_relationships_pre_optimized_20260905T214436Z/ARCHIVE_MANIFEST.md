# Pre-optimization two-level relationship campaign archive

Archived at 2026-09-05 21:44:36 UTC before changing the deterministic
two-level trajectory implementation for exact performance optimization.

This archive is retained for provenance only. Its contents must not be mixed
with, resumed into, or presented as part of the final optimized campaign.

- Model: `mot_simple` deterministic effective two-level MOT.
- Geometry: 25 full-sphere direction discs x 25 uniform-area points, 15 mm
  sampling-disc radius, seed 20260903.
- Common geometry SHA-256:
  `02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17`.
- Force stage: completed on the requested 111-point detuning grid.
- Raw-saturation point `s0=0.25`: completed 625/625 trajectories; loading
  rate 3447.88914707412 atoms/s; independently reconstructed before archival.
- Raw-saturation point `s0=0.5`: interrupted after the 175-ray checkpoint.
- No temperature campaign was run or requested in the final scope.

The restart was intentional: profiling identified mathematically equivalent
field-mask comparisons and multi-ray array batching that preserve every
trajectory, timestep, force evaluation, trapping/escape event, and audit
criterion while substantially reducing wall time. Because those source edits
change the recorded campaign signature, a clean campaign is required.

The original statistics and figures trees are stored separately beneath this
directory. All final outputs will be regenerated from a new, internally
consistent source signature.
