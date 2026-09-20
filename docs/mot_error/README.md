# Archived erroneous multilevel MOT documentation

These files document the former multilevel MOT implementation. They are kept
for provenance only.

The implementation used a transition-specific **saturated two-level scattering
rate** as a bidirectional stimulated transition rate in a multilevel
population-rate matrix. The two-level saturation already encodes the closed
two-state population response, so reusing it as the elementary multilevel rate
is not a valid physical population-rate model. Consequently, forces,
trajectories, capture/loading results, and temperature estimates produced by
this model are not valid multilevel-MOT predictions.

The replacement work belongs in `docs/mot_multilevel` and
`src/pmot/mot_multilevel`.
