# pMOT Geometry and Trapping-Beam Design Record

## Status and authority

The physical pseudo-magneto-optical-trap (pMOT) model has not been
implemented. The current `src/pmot/pmot` package preserves pMOT-owned paths,
differential-polarizability utilities, notebook plotting and trajectory
diagnostics, and an explicitly preliminary two-level scattering helper. It
does **not** yet contain a 1529 nm trapping-beam builder, a scalar/vector/tensor
AC Stark Hamiltonian, a fictitious-field calculation, a trapping-light force,
or an integration of those effects with the production 24-state multilevel
MOT solver.

The authoritative starting point for future pMOT dynamics is the validated
rate-equation implementation in `src/pmot/mot_multilevel`. The preliminary
scattering code in `src/pmot/pmot/preliminary_scattering.py` exists only to
keep the exploratory notebooks reproducible and must not be promoted into the
production pMOT engine.

This document separates three different things:

1. the current, implemented 780 nm MOT-light geometry;
2. a historical 1529 nm proposal recovered from Git history; and
3. physical inferences and open design questions.

Only the first item is current implementation. The recovered proposal and the
inferences are design evidence, not authoritative pMOT requirements.

## Current 780 nm MOT-light geometry

The shared apparatus model uses three mutually orthogonal Cartesian paths:

| Path | Propagation axis | Cell-incidence metadata |
|---|---:|---:|
| `horizontal_x` | x | 45 degrees |
| `horizontal_y` | y | 45 degrees |
| `vertical_z` | z | 0 degrees; enters from above |

Each path has incident and counterpropagating components. Cooling and repump
light share the same six propagation directions, for a total of six cooling
and six repump components. Their reference positions are at the origin. The
shared field layer represents each component with an ideal Gaussian profile
whose nominal radius is 6.35 mm; at 780 nm its Rayleigh range is much longer
than the modeled cell, so this is effectively the current 12.7 mm-diameter
collimated-beam approximation.

The shared apparatus defaults are 20 mW per cooling component, -15 MHz cooling
detuning, 0.5 mW per repump component, zero repump detuning, and 12.7 mm
diameter for both families. Individual studies can and do replace these
defaults. In particular, the August 2026 multilevel loading studies used
27 mW per cooling component and 0.1 mW per repump component. A pMOT
configuration must record its selected baseline rather than silently inheriting
either set of numbers.

Polarization labels follow the propagation-frame convention: `sigma+` and
`sigma-` are defined while looking along each beam's own wavevector. The
generic shared beam builder labels incident and retro components `sigma+` and
`sigma-`, respectively. The production multilevel MOT intentionally replaces
those generic labels with its quadrupole-compatible choices: `sigma+` on the x
and y paths and `sigma-` on the z path, applied in the propagation frame to
both directions on a path. Future pMOT code must specify trapping-light
polarization independently and must not infer it from either convenience
default.

The current beam geometry describes ordinary MOT cooling and repumping. It is
not a model of the focused pMOT trapping light.

## Historical 1529 nm proposal

The initial repository commit, `914a7fc`, contained an exploratory design in
`src/pmot/configuration.py` and `src/pmot/beams.py`. Commit `60c88b3` removed
that design metadata when development pivoted to cooling and repump MOT
trajectories. The following values are recovered verbatim or directly derived
from the initial commit; they are retained here so the design intent is not
lost.

### Recovered values

| Quantity | Historical value |
|---|---:|
| Trap tone 1 wavelength | 1529.376949 nm |
| Trap tone 2 wavelength | 1529.358429 nm |
| Tone 1 relative intensity | 0.492762 |
| Tone 2 relative intensity | 0.507238 |
| Input beam diameter | 35 mm |
| Incident focus offset | -10 mm along its path |
| Retro focus offset | +10 mm along its path |
| Focus separation | 20 mm |
| Mirror-gap metadata | 40 mm |
| Total power per beam pair | 0.5 W |
| Lens identifier | AC508-080-C |
| Lens focal/effective focal length | 80.3 mm |
| Lens back focal length | 66.9 mm |
| Glass-cell outer diameter | 30 mm |
| Glass-cell wall thickness | 5 mm |

The two tones are separated by approximately 2.37377 GHz. The historical
three path labels and code-level unit vectors were

\[
\begin{aligned}
\hat{e}_{\mathrm{oblique\_x}} &= (1,0,1)/\sqrt{2},\\
\hat{e}_{\mathrm{oblique\_y}} &= (0,1,1)/\sqrt{2},\\
\hat{e}_{\mathrm{normal\_z}} &= (0,0,1).
\end{aligned}
\]

Those vectors are not mutually orthogonal:
\(\hat{e}_{\mathrm{oblique\_x}}\cdot
\hat{e}_{\mathrm{oblique\_y}}=1/2\), and each oblique vector has dot product
\(1/\sqrt{2}\) with `normal_z`. This conflicts with both the historical text
calling one path normal to the other two and the present Cartesian x/y/z
model. The historical vectors therefore cannot be adopted without confirming
the real laboratory coordinates.

The initial notebooks explicitly described the implemented field calculation
as a 780 nm cooling-beam check to be completed before adding 1529 nm trapping
physics. No historical code actually constructed the two 1529 nm tones or
applied their Stark shifts. In particular, the old focus geometry was used in
a preliminary cooling-beam visualization; it was not a completed pMOT force
model.

### Differential-polarizability evidence

The narrow Arora CCSD and ARC tables under `data/raw/pmot` span
1529.124--1529.476 nm and contain a sharp dispersive structure near the two
historical tones. The pMOT polarizability utility converts the tabulated values
to MHz per `mW/(100 um)^2` and linearly interpolates within a selected table.

Using the default Arora CCSD table and the current conversion convention gives
the following interpolated coefficients:

| Tone | Scalar | Vector | Tensor |
|---|---:|---:|---:|
| 1529.376949 nm | -738.470536 | -662.817454 | -74.038857 |
| 1529.358429 nm | +717.471334 | +648.808021 | +72.006236 |

Applying the historical intensity fractions gives

\[
\alpha_{\mathrm{weighted}}
=0.492762\,\alpha_1+0.507238\,\alpha_2,
\]

with weighted scalar, vector, and tensor coefficients of approximately
0.038506, 2.488829, and 0.040764 in the same units. Thus the historical pair
appears designed to cancel most of the scalar and tensor differential shifts
while leaving a vector residual roughly 60 times larger than either residual
scalar or tensor coefficient. This interpretation is an inference from the
stored wavelengths, fractions, and tables; the repository contains no
authoritative derivation of the power split.

The generic diffraction-limited Gaussian formula retained in `src/pmot/beams.py`
would give, for a 35 mm input diameter and 80.3 mm focal length, an ideal 1529 nm
waist radius of approximately 2.234 micrometers and a Rayleigh range of
approximately 10.25 micrometers. At a point 10 mm from such a waist, the ideal
Gaussian radius would be approximately 2.179 mm. These are derived ideal-beam
numbers, not measured beam parameters or validated pMOT inputs.

## Inferred trapping mechanism

The historical displacement of the incident and retro waists suggests a
spatial-gradient design. For a symmetric pair with waists on opposite sides of
the origin, the ordinary sum of the two intensities is even about the center,
so its first derivative vanishes there. Their signed intensity difference is
odd and is locally linear. If the two propagation directions carry the
appropriate opposite vector-Stark signs, their helicity-weighted shifts can
therefore produce a zero-crossing, approximately linear state-dependent shift.
That shift could play the role occupied by the Zeeman gradient in a normal MOT.

The two 1529 nm frequencies would then suppress unwanted common differential
scalar and tensor shifts through their opposite dispersive coefficients while
retaining a smaller vector component. Three beam pairs could, in principle,
supply three-dimensional spatially varying vector shifts while the 780 nm
cooling and repump beams provide dissipative radiation pressure. This is the
qualitative mechanism implied by the archived parameters and data; it is not
yet demonstrated by the simulation.

The future solver must distinguish at least three effects of the trapping
light: state-dependent transition shifts that modify the cooling/repump
scattering rates, conservative forces from gradients of the dressed-state
energies, and stochastic heating or loss from trap-light scattering. Treating
the vector shift as a fictitious magnetic field may be a useful representation,
but the actual scalar/vector/tensor Hamiltonian and its action on every
hyperfine state must be defined first.

## Decisions required before construction

The historical proposal is insufficient to specify a production simulation.
The following decisions require experimental input or a documented physical
derivation.

### Laboratory geometry

- Confirm the true lab-frame propagation vectors and the meaning of each
  45-degree cell angle of incidence.
- Decide whether refraction at the glass interfaces changes the in-cell path
  directions and focus positions.
- Define the origin and the signed focus offsets in laboratory coordinates.
- Explain what the 40 mm mirror gap measures and place the lenses, cell
  surfaces, and mirrors explicitly.
- Specify the vertical-path optics; the recovered lens was described as a
  horizontal-path achromat.
- Supply measured or intended beam quality, waist, astigmatism, clipping, and
  aberration parameters, or explicitly authorize the ideal Gaussian model.

### Powers, frequencies, and polarizations

- Define whether 0.5 W is the total before or after the two-tone split, whether
  it is shared between incident and retro components, and whether it is a
  per-path or apparatus-wide quantity.
- Record transmission, retroreflection efficiency, and any imbalance.
- Confirm both wavelengths and their uncertainties, the physical transition
  they straddle, and whether the stored relative intensities are powers at the
  atoms.
- Decide whether the two tones add incoherently or whether their 2.374 GHz beat
  and any coherent cross terms must be retained.
- Specify a complex lab-frame polarization vector for every trapping
  component, including window and mirror transformations. Use the project's
  propagation-frame `sigma+`/`sigma-` convention rather than ambiguous RCP/LCP
  labels.
- Decide whether standing-wave interference is intentionally suppressed or
  must be modeled.

### Atomic Hamiltonian and force

- Document the provenance, raw SI units, state definitions, and sign
  conventions of each polarizability table.
- Map scalar, vector, and tensor shifts onto all 8 ground and 16 excited
  hyperfine-Zeeman states. A fine-structure differential coefficient alone is
  not a hyperfine-resolved Hamiltonian.
- Define whether the pMOT code adds transition-specific Stark shifts directly
  or constructs an effective-field representation, and prove the equivalence
  in the regime being used.
- Define the local quantization basis at and near the fictitious-field zero,
  including any residual real bias field and nonadiabatic state mixing.
- Include conservative forces from spatial derivatives of the dressed-state
  energies where physically required.
- Add imaginary polarizability or another validated calculation for
  trap-light scattering, heating, and loss; the current CSV files contain only
  real differential coefficients.
- Establish safe interpolation and exclusion rules around the sharp poles in
  the narrow 1529 nm tables.

### Integration and validation

- Select and record the 780 nm cooling/repump baseline for the pMOT comparison.
- Extend the production 24-state population-rate solver rather than the
  preliminary two-level notebook helper.
- Recompute local trapping intensities, the full Stark Hamiltonian, dressed
  transition frequencies and polarizations, populations, scattering, and
  force at every required trajectory evaluation.
- Verify even scalar and odd vector symmetry at the origin, the intended
  three-axis restoring signs, and reversal under helicity, focus-gradient, and
  tone-order reversal.
- Demonstrate convergence in field grids, trajectory timesteps, durations,
  and state-basis treatment, followed by representative comparison with the
  event-driven multilevel engine.

Until these decisions are recorded, the historical values should be treated
as a useful reconstruction of an earlier concept, not as defaults for a new
pMOT simulation.
