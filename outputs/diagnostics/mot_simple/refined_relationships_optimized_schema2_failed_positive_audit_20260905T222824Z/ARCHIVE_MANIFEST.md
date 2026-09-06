# Optimized schema-2 campaign archive: positive-boundary audit stop

This campaign attempt is retained for diagnostic provenance only. It stopped
fail-closed at 2026-09-05 22:28:24 UTC and must not be resumed into or mixed
with the final schema-3 campaign.

- Model and geometry: deterministic `mot_simple`, 25 full-sphere direction
  discs x 25 uniform-area points, 15 mm disc radius, seed 20260903.
- Geometry SHA-256:
  `02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17`.
- Force sweep: 111/111 detunings completed; every x/y/z numerical convergence
  check passed; all 45 non-timing CSV columns exactly matched the
  pre-optimization force result.
- `s0=0.25`: 625/625 rays completed in 11.6962 minutes. All sample, audit,
  override, spectrum, loading-by-disc, and geometry files were byte-identical
  to the pre-optimization run. Loading rate was 3447.88914707412 atoms/s.
- `s0=0.5`: stopped with 400/625 accepted rays after ray `(15, 3)` retained an
  internal 2.5 m/s timeout in both 200 ms re-searches.
- Exact follow-up diagnosis: both 200 ms searches returned the same definitive
  bracket, 3.0078125 m/s trapped to 3.154296875 m/s escaped, but the evaluated
  2.5 m/s node was still running. At 250 ms it became definitively trapped at
  both 5 and 2.5 microsecond timesteps, through its second core entry at about
  210.44 ms.
- No temperature simulation or plot is part of this campaign.

The final implementation extends unresolved internal positive-boundary nodes
through the same documented 250/400 ms dual-timestep hierarchy used for zero
capture grids. That source-policy change requires a new schema and clean run;
this archive was therefore preserved rather than metadata-migrated.
