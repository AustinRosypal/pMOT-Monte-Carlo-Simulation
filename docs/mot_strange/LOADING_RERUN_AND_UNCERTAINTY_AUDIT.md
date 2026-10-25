# `mot_strange` loading rerun and uncertainty audit

## Decision

All saved `mot_strange` capture cross sections and loading rates generated with
`kappa = -1.9481511022806496e-7 T/(W/m^2)` are invalid.  The corrected mapping is

```text
kappa = -h C_vector / (mu_B G)
```

where the stored frequency-per-intensity coefficient `C_vector` already
contains the factor from `E^2 = 2 I/(c epsilon_0)`.  The corrected default is
approximately `-9.741e-8 T/(W/m^2)`.  Capture classifications are nonlinear,
so old trajectories and loading integrals cannot be repaired by dividing a
reported value by two.

The corrected rerun must use a new output root.  Existing manifests contain the
old coefficient and must remain immutable provenance.

## Statistical uncertainty

For parameter value `theta`, direction disc `d`, and `P` uniform-area points on
that disc, use the direct capture mask `C[d,p](v; theta)` to form

```text
sigma_d(v; theta) = A_disc/P * sum_p C[d,p](v; theta)
L_d(theta) = K * integral sigma_d(v; theta) v^3 exp(-v^2/u^2) dv
```

The direction discs are the independent clusters.  For `N` discs, the loading
estimate and pointwise 95% Student-t interval are

```text
L_bar(theta) = mean_d L_d(theta)
SEM(theta) = std_d(L_d(theta), ddof=1) / sqrt(N)
CI_95(theta) = L_bar(theta) +/- t_(0.975,N-1) SEM(theta)
```

This calculation is repeated independently at every `theta`.  Equal sample
counts or common launch geometry do not imply equal error bars.  When the same
seeded discs are reused across parameter values, comparisons should also report
paired intervals from `L_d(theta) - L_d(theta_reference)`; treating the two
curves as independent throws away the common-random-number benefit.

A direction-cluster bootstrap is a useful distribution-free cross-check:
resample complete direction discs with replacement, recompute the spectrum and
loading integral for every resample, and quote percentile or BCa limits.  A
hierarchical variant may additionally resample points within each selected
disc.  Pointwise intervals should be labeled pointwise; a simultaneous curve
band requires a max-t or simultaneous bootstrap construction.

An all-empty or all-full sampled mask needs a finite-sample boundary interval.
An ordinary t interval and an ordinary bootstrap both collapse spuriously to
zero width in that case.  Use a conservative direction-cluster binomial bound
(or a documented beta-binomial cluster model) for those spectrum points and
report the resulting loading limit separately from an interior t/bootstrap
interval.

## Numerical and model uncertainty

Statistical bars must not absorb the following independent checks:

- timestep: every classification must agree at the production step and half step;
- duration: retained/time-out cases require a longer-duration terminal check;
- velocity quadrature: compare the loading integral on the production and
  refined velocity grids;
- launch aperture: repeat with a larger disc radius until the spectrum and
  loading integral converge;
- physical inputs: vapor density, temperature, beam power/diameter, detuning,
  and field calibration require a separate sensitivity or input-uncertainty
  propagation.

Any unresolved trajectory causes the quantitative loading result to be
withheld.  It is not legitimate to count an indeterminate case as escaped and
hide that choice inside a statistical error bar.

## Rerun acceptance gates

1. New output root and manifest containing the corrected `kappa` and source digest.
2. Full-sphere direction sampling and uniform-area points; no octant multiplier.
3. Direct velocity-resolved capture masks with fail-closed dual-step and
   duration convergence at every unresolved node.
4. Velocity-grid and launch-aperture convergence.
5. Per-disc spectra and per-disc loading integrals retained for independent audit.
6. Per-parameter clustered intervals plus paired-difference intervals for scans.
7. No quantitative result while any required node remains indeterminate.
8. Explicit statement that `mot_strange` is a surrogate-field conventional MOT,
   not a physical pMOT prediction.
