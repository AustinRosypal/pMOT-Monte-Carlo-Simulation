# Two-level versus multilevel loading comparison gate

No overlaid loading-rate figure may be generated until the parameter table for
that figure has been reviewed and approved.  The comparison is intended to
isolate the change from the deterministic closed two-level mean-force model to
the 24-state population-rate model.  Parameters unrelated to that model change
must therefore be identical.

## Required matches

The comparison manifest must record, for both models and for every scan:

- the sampled x grid and the definition of the x variable;
- cooling power per traveling beam, Gaussian 1/e^2 diameter, center intensity,
  and the single-beam `s0 = I0/I_sat` convention;
- cooling detuning in Hz and in linewidth units, including the linewidth used
  for the conversion;
- axial quadrupole gradient and coil geometry;
- cooling-beam propagation directions and propagation-frame polarizations;
- gravity state and recoil-diffusion state;
- launch radius, sampling-disc radius, full-sphere direction distribution,
  points per disc, random seed, and exact saved launch geometry hash;
- trapped/escaped criterion, velocity grid, timestep, duration/audit ladder,
  and treatment of nonmonotone capture masks and the zero-speed node;
- loading-rate normalization, cross-section convention, and absence of an
  octant or 4-pi multiplier; and
- independent statistical cluster, confidence level, and Student-t interval
  construction.

The cooling-light parameters and launch geometry must match point by point,
not merely share nominal labels.  The multilevel-only repump beams and the
internal-state/population equations are intentional model differences and must
be listed explicitly on the figure rather than treated as matched parameters.

## Plot contract after approval

Each approved comparison figure will place the two sampled means on the same
axes and show a separate translucent two-sided 95% Student-t band for each
model.  Direction discs, not individual launch points, are the independent
clusters.  Plotting must fail closed if the manifest contains an unapproved
mismatch or if either dataset has unresolved capture classifications.

## Current readiness status (2026-11-03)

The retained August/September multilevel relationship outputs are provenance
from the removed invalid saturated-per-transition closure.  They cannot be
used as physical multilevel curves or as evidence for the effect of atomic
populations.  A quantitative overlay requires a new matched loading campaign
from the replacement Section-12 population-rate solver after its capture and
loading validation gates pass.  The existing Section-12 magnetic-gradient
campaign is explicitly diagnostic and does not yet meet that requirement.

Even as a schema/provenance check, the current optimization-oriented two-level
campaign is not automatically compatible with the retained September grids:

- its raw/effective-saturation scans use `Delta/Gamma = -2.25`
  (`-13.64625 MHz`), whereas the retained multilevel scans use `-15 MHz`
  (`Delta/Gamma = -2.471169686985173`);
- the current two-level configuration uses `Gamma/(2*pi) = 6.065 MHz`, while
  the earlier multilevel relationship grid used `6.07 MHz`; a matched campaign
  must choose one linewidth and record both the angular and ordinary-frequency
  detunings explicitly;
- the saved full-sphere launch-geometry hashes differ, even though both use the
  nominal seed `20260903` and 25 direction discs by 25 points;
- the two-level detuning scan selects cooling power from its corrected
  saturation scan, whereas the multilevel detuning scan fixes `27 mW` per
  cooling traveling beam; and
- no currently selected multilevel magnetic-gradient or cooling-diameter
  dataset has yet been accepted as a point-for-point match to the new
  two-level grids.

These differences must be resolved by running matched, validated datasets
after parameter review.  They must not be hidden by interpolation or by
overlaying nominally similar legacy plots.  Historical invalid-solver outputs
must never be selected merely because their grids happen to match.
