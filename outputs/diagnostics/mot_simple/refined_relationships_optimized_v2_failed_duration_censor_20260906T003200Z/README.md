# Archived two-level relationship campaign (optimized v2)

This directory preserves the failed optimized-v2 production campaign exactly
as it existed after the fail-closed stop on 2026-09-06 UTC.  It is retained for
provenance and is not a final scientific result.

- Failure point: raw-saturation `s0 = 1`, launch ray `(disc 11, point 14)`.
- Completed before the stop: all 625 rays at `s0 = 0.25`, `0.5`, and `0.75`;
  380 of 625 rays at `s0 = 1`.
- Cause: the 200 ms dual-timestep searches both censored the same boundary node
  at 2.421875 m/s.  Independent 250 ms searches later showed capture at about
  209.84 ms and recovered a clean, timestep-consistent bracket.
- Disposition: superseded by a new schema and output root that adds a complete
  longer-duration boundary fallback and an authoritative direct velocity-grid
  guard for any exceptional positive-threshold ray.

The `statistics` and `figures` subdirectories are the original campaign roots.
Do not combine their partial aggregates with the replacement campaign.
