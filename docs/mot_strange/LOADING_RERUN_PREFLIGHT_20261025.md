# Corrected strange-MOT loading preflight, 2026-10-25

This records the requested gradient magnitudes (10,10,20) G/cm and the
inside-cell rejection-sampling change. No loading trajectories were run for
this preflight. Existing results and default historical field powers remain
unchanged; the new calibration is an explicit configuration.

## Field conversion and optical geometry

The conversion is

```text
B = kappa sum_b(helicity_b I_b k_hat_b)
kappa = -h C_vector/(mu_B G) = -9.740755511403248e-8 T/(W/m^2)
C_vector = 1363.34 Hz/(W/m^2), G = 1
```

The factor from E^2=2I/(c epsilon_0) is already in C_vector. No additional
factor of two is applied. The surrogate directly uses intensity differences;
the field gradient follows by differentiating those intensity profiles.

This calibration specifically uses the existing mot_strange surrogate's
1530.000 nm wavelength, forward/return waist positions -10.0326971 and
+9.97601576 mm, and Rayleigh ranges 9.1030720913 and 9.0005040185 micrometers.
It must not be silently mixed with the separate 1529.268881 nm ideal pMOT
geometry. The surrogate model assumes equal incident and return powers, with
no round-trip losses. Its forward and return waist radii are respectively
2.10554618 and 2.09365054 micrometers.

For an axial component with waist z_w and Rayleigh range z_R,

```text
I(q) = 2 P z_R/[lambda ((q-z_w)^2+z_R^2)]
dI/dq at q=0 = 4 P z_R z_w/[lambda (z_w^2+z_R^2)^2]
```

Calibrate each path power using these analytic derivatives. The configured
helicity signs give signed central gradients (+10,+10,-20) G/cm, or
(+0.1,+0.1,-0.2) T/m. Independent central differences at 1 micrometer,
0.1 micrometer, and 0.01 micrometer verify the full 3D Jacobian, including
zero off-diagonal entries. These are local origin gradients, not constant
gradients throughout the cell.

| Path | Power of each traveling component (mW) | Origin intensity of each traveling component (W/cm^2) | Signed dB_i/dx_i (G/cm) |
| --- | ---: | ---: | ---: |
| x | 21.71913896 | 0.2567635622 | +10 |
| y | 21.71913896 | 0.2567635622 | +10 |
| z | 43.43827791 | 0.5135271244 | -20 |

The ideal three-path incident power budget is 86.87655583 mW with a 1:1:2
split. The retro pass reuses that light; it is not another independent source
power requirement. The incoherent sum of all six component intensities at the
origin is 2.054108498 W/cm^2. Each x/y component peaks at its own waist at
3.11884065e9 W/m^2 (forward) or 3.15438238e9 W/m^2 (return); z values are twice
these. Do not confuse the waist peaks with the origin intensities.

This is an equivalent-field calibration within mot_strange. It is not a
validated physical multilevel pMOT power prediction or an uncertainty estimate
for the stored vector-polarizability coefficient.

## Existing vapor normalization

The live loading calculation uses

```text
R = K integral sigma(v) v^3 exp(-v^2/u^2) dv
K = 911960
u^2 = 56670 m^2/s^2
```

With the repository's Rb-87 mass, u^2=2 k_B T/m implies T=296.17928934 K,
or 23.02928934 degrees C. The normalized Maxwell speed distribution is

```text
f_MB(v) = A v^2 exp(-v^2/u^2)
A = 4/(sqrt(pi) u^3) = 1.67284320275e-7 s^3/m^3
R = n87 integral sigma(v) v f_MB(v) dv
```

Therefore the existing numerical rate formula is equivalent to
n87=K/A=5.45155695704e12 m^-3 (=5.45155695704e6 cm^-3).
Its Rb-87 partial pressure is 2.22924870640e-8 Pa, or
1.67207403589e-10 Torr. This is not the total vacuum pressure.
The code does not specify a measured cell density or uncertainty, isotope
abundance, or a saturated-vapor equilibrium constraint.

There is a separate inconsistency in the distribution written in saved
metadata: the historical coefficient 2.80e-7 makes that distribution integrate
to 1.67379703931 instead of one. Factoring K using that coefficient would
suggest n87=3.257e12 m^-3, but this density and the stated thermal scale do not
reproduce the existing rate with a normalized Maxwell distribution. Keep the
actual numerical formula separate from that historical interpretation.
Neither an intended physical density nor a calibrated experimental vapor
condition can be recovered uniquely from these two inconsistent coefficients.

Do not silently change K or the thermal scale. Report the density-independent
coefficient R/n87 using the normalized distribution, and label any retained
atoms/s value with its assumed n87. A measured Rb-87 absorption density and
vapor temperature would establish an experimental normalization; a total
vacuum-gauge pressure alone cannot supply the Rb-87 partial density.

For context only, the saturated solid-Rb vapor model in
[Steck's Rb-87 data](https://steck.us/alkalidata/rubidium87numbers.pdf)
gives log10(P_Torr)=2.881+4.857-4215/T and specifies better than +/-5%
accuracy over 298--550 K. Combined with natural Rb-87 abundance 0.2783,
at 25 degrees C this would give n87 about 3.60e15 m^-3, roughly 660 times
the retained normalization. This is not a correction factor or evidence that
the experimental MOT cell is saturated. A pumped/dispenser-fed cell need not
be in saturated-vapor equilibrium. The existing normalization is an assumption
with unquantified uncertainty, not a verified 5%-accurate experimental input.

## Inside-cell launch sampling and area weighting

Retain uniform full-sphere directions, 15 mm nominal disc radius, and disc
centers 10 mm from the origin. Retain the current cube with half-length
14.1421356 mm, corresponding to a 28.2842712 mm internal side in this model.
The physical glass-cell dimensions have not been independently verified here.

For each fixed direction disc, draw uniform-area proposals, reject points
outside the cube, and retry until there are exactly P accepted points. Do not
reject directions or rotate them to favor a larger intersection. Every
velocity remains parallel to the incident normal, at every impact parameter.

The accepted points are uniform on the intersection D_d of that disc and the
cell, not the complete circular disc. The correct cross-section estimator is

```text
sigma_d(v) = area(D_d)/P sum_p C[d,p](v)
sigma(v) = mean_d sigma_d(v)
```

Using pi*(15 mm)^2 for a clipped disc would overestimate its contribution.
Compute area(D_d) by deterministic angular quadrature with explicit geometric
breakpoints, not from a noisy rejection fraction. Preserve each direction's
area in the launch manifest and trajectory rows. The estimator concerns these
specified interior launch sections; it does not by itself prove convergence
with launch-plane placement or inclusion of every possible cell-entry ray.
The 15 mm aperture and launch placement still need convergence checks before
claiming a total cell loading rate.

## Precision and production gates

Use a 95% direction-cluster interval with relative half-width target <=5%.
An independent pilot determines a fixed production allocation before
production; do not repeatedly enlarge a sample until an ordinary interval
happens to pass. Numerical convergence and vapor/model uncertainties remain
separate. Retain the fail-closed gates in LOADING_RERUN_AND_UNCERTAINTY_AUDIT.md;
the older representative-only trajectory audits do not establish production
convergence. Central-core loading is primary, with off-origin retention
reported separately.

Run `uv run python scripts/prepare_mot_strange_loading_rerun.py` to generate
the calibrated configuration, six-component intensity table, vapor audit,
source digests, and a 32-disc by 20-point geometry preview. Its single 1 m/s
placeholder only materializes each launch once: it is not a production speed
grid and does not run any dynamics. Outputs go to the separately named
`outputs/diagnostics/mot_strange/loading_rerun_10_10_20Gpcm_20261025_preflight`.
